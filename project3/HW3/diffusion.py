import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    image_size: int = 32
    channels: int = 1
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, config: DiffusionConfig):
        super().__init__()
        self.model = model
        self.config = config

        T = config.timesteps
        betas = torch.linspace(config.beta_start, config.beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Register buffers for use on correct device
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        # posterior variance Var[q(x_{t-1}|x_t, x_0)] with alpha_bar_{-1} defined as 1 for t=0
        posterior_var = torch.empty_like(betas)
        posterior_var[0] = 0.0
        posterior_var[1:] = betas[1:] * (1.0 - alpha_bars[:-1]) / (1.0 - alpha_bars[1:])
        self.register_buffer("posterior_variance", posterior_var)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Draw x_t ~ q(x_t | x_0) = N(sqrt(alpha_bar_t) x0, (1 - alpha_bar_t) I)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and variance of p(x_{t-1} | x_t)
        using epsilon prediction parameterization.
        """
        # Support optional conditioning vector (e.g., CLIP text embedding)
        eps_pred = self.model(x_t, t, cond) if cond is not None else self.model(x_t, t)
        a_t = self.alphas[t].view(-1, 1, 1, 1)
        a_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)

        # x0_pred from eps
        x0_pred = (x_t - torch.sqrt(1 - a_bar_t) * eps_pred) / torch.sqrt(a_bar_t)
        # mean of q(x_{t-1}|x_t,x0)
        # compute alpha_bar_{t-1} per element (define as 1 when t==0)
        a_bar_prev = torch.where(
            (t > 0).view(-1, 1, 1, 1),
            self.alpha_bars[(t - 1).clamp_min(0)].view(-1, 1, 1, 1),
            torch.ones_like(a_bar_t),
        )
        mean = (
            torch.sqrt(a_bar_prev) * beta_t / (1.0 - a_bar_t) * x0_pred
            + torch.sqrt(a_t) * (1.0 - a_bar_prev) / (1.0 - a_bar_t) * x_t
        )
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        return mean, var

    @torch.no_grad()
    def sample_ddpm(self, batch_size: int, device: torch.device, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """DDPM ancestral sampling over all timesteps."""
        img = torch.randn(batch_size, self.config.channels, self.config.image_size, self.config.image_size, device=device)
        for t_int in reversed(range(self.config.timesteps)):
            t = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
            eps = self.model(img, t, cond) if cond is not None else self.model(img, t)
            a_t = self.alphas[t].view(-1, 1, 1, 1)
            a_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1, 1)

            mean = (img - (1 - a_t) / torch.sqrt(1 - a_bar_t) * eps) / torch.sqrt(a_t)
            if t_int > 0:
                noise = torch.randn_like(img)
                var = self.posterior_variance[t].view(-1, 1, 1, 1)
                img = mean + torch.sqrt(var) * noise
            else:
                img = mean
        return img

    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size: int,
        device: torch.device,
        steps: int = 50,
        eta: float = 0.0,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DDIM sampling with uniform time stride. When eta=0, deterministic.
        """
        T = self.config.timesteps
        assert steps >= 1 and steps <= T
        # choose timesteps uniformly
        ts = torch.linspace(0, T - 1, steps, dtype=torch.long, device=device)
        img = torch.randn(batch_size, self.config.channels, self.config.image_size, self.config.image_size, device=device)
        for i in reversed(range(steps)):
            t = ts[i].expand(batch_size)
            a_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

            ###################################### DDIM Task ######################################
            eps_pred = self.model(img, t, cond) if cond is not None else self.model(img, t)
            x0_pred = (img - torch.sqrt(1.0 - a_bar_t) * eps_pred) / torch.sqrt(a_bar_t)

            if i == 0:
                # 最后一步直接到 x0
                img = x0_pred
                break

            # 上一个子时间步（而非 t-1）
            t_prev = ts[i - 1].expand(batch_size)
            a_bar_prev = self.alpha_bars[t_prev].view(-1, 1, 1, 1).clamp(1e-12, 1.0)

            # DDIM sigma：eta * sqrt((1-ā_prev)/(1-ā_t) * (1 - ā_t/ā_prev))
            sigma_t = eta * torch.sqrt(
                (1.0 - a_bar_prev) / (1.0 - a_bar_t) * (1.0 - a_bar_t / a_bar_prev)
            ).clamp_min(0.0)

            # 均值项：确定性部分 + 残差项（与 eps_pred 同向）
            # 注意：当 eta=0 时，第二项与第三项（噪声）都应消失
            c = torch.sqrt(1.0 - a_bar_prev - sigma_t ** 2).clamp_min(0.0)
            mean = torch.sqrt(a_bar_prev) * x0_pred + c * eps_pred            
            ###################################### DDIM Task ######################################
        return img

    def training_loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x0.size(0)
        if noise is None:
            noise = torch.randn_like(x0)
        t = torch.randint(0, self.config.timesteps, (B,), device=x0.device, dtype=torch.long)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.model(x_t, t, cond) if cond is not None else self.model(x_t, t)
        return F.mse_loss(eps_pred, noise)
