"""UNet: (B, 8, 72, 48) → (B, 3, 72, 48).

4 段（72×48 → 36×24 → 18×12 → 9×6）。各段 [Conv3×3 → GroupNorm → GELU] × 2、down は MaxPool2、
up は双一次 2 倍 + skip 結合。出力は 1×1 conv（活性化なし。正規化済み場は符号を持つ）。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 8,
        out_ch: int = 3,
        widths: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.widths = tuple(widths)
        self.enc = nn.ModuleList()
        cin = in_ch
        for w in widths:
            self.enc.append(Block(cin, w))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.dec = nn.ModuleList()
        for w_skip, w_up in zip(widths[-2::-1], widths[:0:-1], strict=True):
            self.dec.append(Block(w_up + w_skip, w_skip))
        self.head = nn.Conv2d(widths[0], out_ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        skips: list[Tensor] = []
        for k, block in enumerate(self.enc):
            x = block(x)
            if k + 1 < len(self.enc):
                skips.append(x)
                x = self.pool(x)
        for block in self.dec:
            s = skips.pop()
            x = nn.functional.interpolate(
                x, size=s.shape[-2:], mode="bilinear", align_corners=False
            )
            x = block(torch.cat([x, s], dim=1))
        return self.head(x)
