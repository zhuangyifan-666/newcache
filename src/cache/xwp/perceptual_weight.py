"""Perceptual frequency weighting for xWPCache-v2 diagnostics."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .frequency import fft_filter_2d, normalize_filter_mean, radial_frequency_grid


def q_noise_gate(t: float | torch.Tensor, gate_start: float = 0.3):
    """Noise gate ``clip((t - gate_start) / (1 - gate_start), 0, 1)``."""

    if isinstance(t, torch.Tensor):
        return ((t.to(torch.float32) - float(gate_start)) / max(1e-8, 1.0 - float(gate_start))).clamp(0.0, 1.0)
    return max(0.0, min(1.0, (float(t) - float(gate_start)) / max(1e-8, 1.0 - float(gate_start))))


def perceptual_frequency_weight(
    t: float | torch.Tensor,
    size: int = 64,
    lambda_d: float = 0.7,
    lambda_l: float = 0.3,
    sigma_d: float = 0.25,
    sigma_l: float = 0.15,
    sigma_h: float = 0.85,
    normalize_mean: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the timestep-aware perceptual frequency weight ``W_t(f)``."""

    if isinstance(t, torch.Tensor):
        device = t.device if device is None else device
    freq = radial_frequency_grid(int(size), int(size), device=device, dtype=dtype, normalize=True).to(torch.float32)
    wd = torch.exp(-freq.square() / (float(sigma_d) ** 2))
    wl = (1.0 - torch.exp(-freq.square() / (float(sigma_l) ** 2))) * torch.exp(
        -freq.pow(4) / (float(sigma_h) ** 4)
    )
    q = q_noise_gate(t)
    if isinstance(q, torch.Tensor) and q.ndim > 0:
        q_view = q.flatten().to(device=freq.device, dtype=torch.float32).view(-1, 1, 1)
        weight = float(lambda_d) * wd.view(1, *wd.shape) + float(lambda_l) * q_view * wl.view(1, *wl.shape)
        weight = weight.unsqueeze(1)
    else:
        q_float = float(q.item()) if isinstance(q, torch.Tensor) else float(q)
        weight = float(lambda_d) * wd + float(lambda_l) * q_float * wl
    if normalize_mean:
        weight = normalize_filter_mean(weight)
    return weight


def _downsample(x: torch.Tensor, size: int) -> tuple[torch.Tensor, bool]:
    if x.ndim == 3:
        x_b = x.unsqueeze(0)
        squeezed = True
    elif x.ndim == 4:
        x_b = x
        squeezed = False
    else:
        raise ValueError(f"Expected x shape [C,H,W] or [B,C,H,W], got {tuple(x.shape)}")
    if int(size) > 0 and x_b.shape[-2:] != (int(size), int(size)):
        x_b = F.interpolate(x_b.to(torch.float32), size=(int(size), int(size)), mode="bilinear", align_corners=False)
    else:
        x_b = x_b.to(torch.float32)
    return x_b, squeezed


def phi_perceptual(
    x: torch.Tensor,
    t: float | torch.Tensor,
    size: int = 64,
    lambda_d: float = 0.7,
    lambda_l: float = 0.3,
    sigma_d: float = 0.25,
    sigma_l: float = 0.15,
    sigma_h: float = 0.85,
    normalize_mean: bool = True,
) -> torch.Tensor:
    """Apply low-resolution perceptual FFT weighting ``Phi_t`` to ``x``."""

    x_small, squeezed = _downsample(x, size)
    weight = perceptual_frequency_weight(
        t,
        size=size,
        lambda_d=lambda_d,
        lambda_l=lambda_l,
        sigma_d=sigma_d,
        sigma_l=sigma_l,
        sigma_h=sigma_h,
        normalize_mean=normalize_mean,
        device=x_small.device,
        dtype=torch.float32,
    )
    out = fft_filter_2d(x_small, weight, norm="ortho")
    return out.squeeze(0) if squeezed else out

