import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    A tiny CNN that maps a 1x32x32 MNIST image to an embedding vector of size embed_dim.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(128, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 32, 32)
        h = self.net(x)  # (B, 128, 4, 4)
        h = h.mean(dim=[2, 3])  # GAP -> (B, 128)
        z = self.proj(h)  # (B, D)
        z = F.normalize(z, dim=-1)  # unit-norm embeddings
        return z


class TextEncoder(nn.Module):
    """
    Simplest text encoder: learnable embedding per class (0..9) for MNIST.
    """

    def __init__(self, num_classes: int = 10, embed_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(num_classes, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        # labels: (B,) int64 in [0, num_classes)
        z = self.emb(labels)
        z = F.normalize(z, dim=-1)
        return z

    def all_class_embeddings(self) -> torch.Tensor:
        # (C, D)
        return F.normalize(self.emb.weight, dim=-1)


class SimpleCLIP(nn.Module):
    """
    A minimal CLIP-style model for MNIST that learns aligned embeddings
    for images and class texts (digits 0..9) with a temperature.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.image_encoder = ImageEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.text_encoder = TextEncoder(num_classes=num_classes, embed_dim=embed_dim)
        # logit_scale initialized to log(1/0.07) as in CLIP
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(x)

    def forward_text(self, labels: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(labels)

    def compute_logits(self, image_embs: torch.Tensor, class_embs: torch.Tensor) -> torch.Tensor:
        # image_embs: (B, D), class_embs: (C, D)
        # logits: (B, C)
        logit_scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        return logit_scale * (image_embs @ class_embs.t())

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        img_z = self.forward_image(x)
        class_z = self.text_encoder.all_class_embeddings()
        logits = self.compute_logits(img_z, class_z)
        return logits.argmax(dim=-1)
