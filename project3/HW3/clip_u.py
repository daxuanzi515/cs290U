#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic CLIP for MNIST
A lightweight CLIP that learns image–text alignment with semantic initialization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================
# Image Encoder
# ==========================
class ImageEncoder(nn.Module):
    """Tiny CNN image encoder for 1×32×32 MNIST."""
    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),  # 16×16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),  # 8×8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),  # 4×4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(128, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.net(x)
        h = h.mean(dim=[2, 3])
        z = self.proj(h)
        return F.normalize(z, dim=-1)


# ==========================
# Text Encoder (semantic)
# ==========================
class TextEncoder(nn.Module):
    """
    Generate semantic text embeddings for digits 0–9.
    - Each class has a deterministic semantic vector as an anchor.
    - A learnable linear projection adapts it to the shared CLIP space.
    """
    def __init__(self, num_classes=10, embed_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. Create semantic anchors
        torch.manual_seed(42)
        anchors = torch.randn(num_classes, embed_dim)
        anchors = F.normalize(anchors, dim=-1)
        self.register_buffer("anchors", anchors)  # non-trainable

        # 2. Learnable adaptation
        self.adapter = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.adapter.weight)

    def forward(self, labels: torch.Tensor):
        """labels: (B,) int tensor"""
        z = self.anchors[labels]
        z = self.adapter(z)
        return F.normalize(z, dim=-1)

    def all_class_embeddings(self):
        """Return adapted prototypes (C, D)"""
        z = self.adapter(self.anchors)
        return F.normalize(z, dim=-1)


# ==========================
# Semantic CLIP
# ==========================
class SemanticCLIP(nn.Module):
    """
    CLIP-style contrastive model for MNIST.
    Learns alignment between CNN image embeddings and semantic digit embeddings.
    """
    def __init__(self, in_channels=1, embed_dim=128, num_classes=10):
        super().__init__()
        self.image_encoder = ImageEncoder(in_channels, embed_dim)
        self.text_encoder = TextEncoder(num_classes, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward_image(self, x):
        return self.image_encoder(x)

    def forward_text(self, labels):
        return self.text_encoder(labels)

    def compute_logits(self, img_embs, txt_embs):
        scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        return scale * img_embs @ txt_embs.T

    def forward(self, images, labels):
        img_emb = self.forward_image(images)
        txt_emb = self.forward_text(labels)
        return img_emb, txt_emb

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        img_z = self.forward_image(x)
        class_z = self.text_encoder.all_class_embeddings()
        logits = self.compute_logits(img_z, class_z)
        return logits.argmax(dim=-1)
