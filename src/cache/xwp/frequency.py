"""Frequency-domain helpers for xWPCache-v2 diagnostics."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def radial_frequency_grid(
    height: int,
    width: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
) -> torch.Tensor:
    """Return an FFT-ordered radial frequency grid with shape ``[H, W]``."""

    fy = torch.fft.fftfreq(int(height), device=device, dtype=dtype).view(height, 1)
    fx = torch.fft.fftfreq(int(width), device=device, dtype=dtype).view(1, width)
    freq = torch.sqrt(fx.square() + fy.square())
    if normalize:
        max_radius = math.sqrt(0.5**2 + 0.5**2)
        freq = freq / max_radius
    return freq


def natural_image_spectrum(
    freq: torch.Tensor,
    beta: float = 2.0,
    f0: float = 0.03,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Natural-image power spectrum prior ``(f^2 + f0^2)^(-beta/2)``."""

    base = freq.to(torch.float32).square() + float(f0) ** 2
    return base.clamp_min(float(eps)).pow(-0.5 * float(beta))


def normalize_filter_mean(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a spatial frequency filter by its mean over the last two dims."""

    return weight / weight.mean(dim=(-2, -1), keepdim=True).clamp_min(float(eps))


def _as_bchw(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim == 4:
        return x, False
    raise ValueError(f"Expected x shape [C,H,W] or [B,C,H,W], got {tuple(x.shape)}")


def _prepare_weight(weight: torch.Tensor, batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    if weight.ndim == 2:
        weight = weight.view(1, 1, height, width)
    elif weight.ndim == 4:
        if weight.shape[-2:] != (height, width):
            raise ValueError(f"Filter shape {tuple(weight.shape)} does not match H,W={(height, width)}")
    else:
        raise ValueError(f"Expected filter shape [H,W], [1,1,H,W], or [B,1,H,W], got {tuple(weight.shape)}")

    if weight.shape[0] not in {1, batch}:
        raise ValueError(f"Filter batch {weight.shape[0]} is incompatible with input batch {batch}")
    if weight.shape[1] != 1:
        raise ValueError("Filter channel dimension must be 1 for broadcasting over image channels")
    return weight.to(device=device, dtype=torch.float32)


def fft_filter_2d(x: torch.Tensor, weight: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Apply a real FFT-domain filter to ``x`` and return a real tensor."""

    x_bchw, squeezed = _as_bchw(x)
    batch, _channels, height, width = x_bchw.shape
    filt = _prepare_weight(weight, batch, height, width, x_bchw.device)
    spectrum = torch.fft.fftn(x_bchw.to(torch.float32), dim=(-2, -1), norm=norm)
    filtered = torch.fft.ifftn(spectrum * filt, dim=(-2, -1), norm=norm).real
    return filtered.squeeze(0) if squeezed else filtered

