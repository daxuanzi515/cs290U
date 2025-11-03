import math
from typing import List, Tuple
from torchvision.utils import save_image, make_grid
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 100):
        super().__init__()

        # Encoder: Convolutions to extract features (input: 1x32x32)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 16, 16)
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (B, 128, 8, 8)
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (B, 256, 4, 4)
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(256),
        )

        # Latent space
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder: Transposed convolutions for upsampling
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (B, 128, 8, 8)
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 16, 16)
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # -> (B, 1, 32, 32)
            nn.Sigmoid(),  # Pixel values in [0, 1]
        )

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @torch.no_grad()
    def canonicalize_latent(self, z: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Canonicalize latent vectors by flipping sign so that decoded images prefer black background.
        Rule: if decoded pixel mean > threshold, flip sign of z (z -> -z).
        This breaks the global sign symmetry in latent space and stabilizes polarity for downstream models.

        z: (B, D) latent vectors
        returns: z' with per-sample sign possibly flipped
        """
        if z.dim() != 2:
            z_flat = z.view(z.size(0), -1)
        else:
            z_flat = z
        dec = self.decode(z_flat)
        mean = dec.mean(dim=(1, 2, 3), keepdim=True)  # (B,1,1,1)
        flip = (mean > threshold).float().view(-1, 1)
        z_aligned = z_flat * (1.0 - 2.0 * flip)
        return z_aligned


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
    loss_type: str = "mse",
):
    """
    VAE loss = reconstruction + kl_weight * KL.

    - loss_type: 'bce' for BCE (since decoder outputs sigmoid probs), 'mse' for MSE.
    Uses reduction='sum' to approximate likelihood objective.
    Returns: total_loss, rec_loss(item), kl_loss(item)
    """
    if loss_type == "mse":
        rec = nn.functional.mse_loss(recon_x, x, reduction="sum")
    else:
        bce = nn.BCELoss(reduction="sum")
        rec = bce(recon_x, x)

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = rec + kl_weight * kl
    return total, rec.item(), kl.item()


@torch.no_grad()
def save_random_samples(model: ConvVAE, device: torch.device, out_dir: Path, epoch: int, n_samples: int = 64, latent_dim: int = 100, nrow: int = 8):
    model.eval()
    z = torch.randn(n_samples, latent_dim, device=device)
    samples = model.decode(z).cpu()
    grid = make_grid(samples, nrow=nrow, padding=2)
    out_path = out_dir / f"samples_epoch_{epoch:03d}.png"
    save_image(grid, out_path)
    return out_path


@torch.no_grad()
def save_reconstructions(model: ConvVAE, device: torch.device, out_dir: Path, epoch: int, batch: torch.Tensor, n_show: int = 64, nrow: int = 8):
    model.eval()
    x = batch[:n_show].to(device)
    recon, _, _ = model(x)
    grid = make_grid(torch.cat([x.cpu(), recon.cpu()], dim=0), nrow=nrow, padding=2)
    out_path = out_dir / f"recon_epoch_{epoch:03d}.png"
    save_image(grid, out_path)
    return out_path
