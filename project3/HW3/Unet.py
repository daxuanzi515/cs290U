import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: (B,) int64 or float tensor of values in [0, T)
    returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
    )  # (half,)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.t_proj = nn.Linear(t_dim, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        # time embedding as bias
        t_bias = self.t_proj(t).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        h = h + t_bias
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    A small UNet for 32x32 grayscale images. Predicts epsilon (noise) given x_t and t.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        time_dim: Optional[int] = None,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        ch = base_channels
        self.in_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        t_dim = time_dim or ch * 4
        self.time_mlp = TimeMLP(in_dim=ch * 4, out_dim=t_dim)
        # Optional conditioning pathway (e.g., CLIP text embedding)
        self.cond_dim = cond_dim
        if cond_dim is not None and cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, t_dim), SiLU(), nn.Linear(t_dim, t_dim)
            )
        else:
            self.cond_proj = None

        # Down path
        in_chs: List[int] = [ch]
        downs = []
        curr_ch = ch
        for mult in channel_mults:
            out_ch = base_channels * mult
            downs.append(ResBlock(curr_ch, out_ch, t_dim))
            downs.append(ResBlock(out_ch, out_ch, t_dim))
            in_chs.append(out_ch)
            downs.append(Downsample(out_ch))
            curr_ch = out_ch
        self.downs = nn.ModuleList(downs)

        # Mid
        self.mid1 = ResBlock(curr_ch, curr_ch, t_dim)
        self.mid2 = ResBlock(curr_ch, curr_ch, t_dim)

        # Up path
        ups = []
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            # upsample first to match the spatial resolution of the corresponding skip
            ups.append(Upsample(curr_ch))
            # then concat with skip (curr_ch + out_ch) and process with resblocks
            ups.append(ResBlock(curr_ch + out_ch, out_ch, t_dim))
            ups.append(ResBlock(out_ch, out_ch, t_dim))
            curr_ch = out_ch
        self.ups = nn.ModuleList(ups)

        self.out_norm = nn.GroupNorm(8, curr_ch)
        self.out_act = SiLU()
        self.out_conv = nn.Conv2d(curr_ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, H, W) in [-1, 1]
        # t: (B,) int or float
        B = x.size(0)
        x = self.in_conv(x)
        # time embedding
        t_emb = sinusoidal_time_embedding(t, dim=self.time_mlp.net[0].in_features)
        t_emb = self.time_mlp(t_emb)

        if cond is not None and self.cond_proj is not None:
            c_emb = self.cond_proj(cond)
            t_emb = t_emb + c_emb

        skips: List[torch.Tensor] = []
        h = x
        # down path: blocks come in triplets [res, res, down]
        for i in range(0, len(self.downs), 3):
            h = self.downs[i](h, t_emb)
            h = self.downs[i + 1](h, t_emb)
            skips.append(h)
            h = self.downs[i + 2](h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # up path mirrors down; pop skips in reverse
        for i in range(0, len(self.ups), 3):
            upsampler = self.ups[i]
            res1 = self.ups[i + 1]
            res2 = self.ups[i + 2]
            skip = skips.pop()
            # upsample only if spatial sizes differ
            if h.shape[2:] != skip.shape[2:]:
                h = upsampler(h)
            h = torch.cat([h, skip], dim=1)
            h = res1(h, t_emb)
            h = res2(h, t_emb)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h
