"""Wiener clean-image proxy for x-prediction PixelGen trajectories."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .frequency import fft_filter_2d, natural_image_spectrum, normalize_filter_mean, radial_frequency_grid


def _t_to_tensor(t: float | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t.to(device=device, dtype=torch.float32)
    return torch.tensor(float(t), device=device, dtype=torch.float32)


def wiener_filter_xpred(
    t: float | torch.Tensor,
    freq_or_spectrum: torch.Tensor,
    beta: float = 2.0,
    f0: float = 0.03,
    eps: float = 1e-8,
    normalize_mean: bool = False,
) -> torch.Tensor:
    """Return ``G_t(f) = t S_x(f) / (t^2 S_x(f) + (1-t)^2 + eps)``."""

    base = freq_or_spectrum.to(torch.float32)
    spectrum = natural_image_spectrum(base, beta=beta, f0=f0, eps=eps) if float(base.max()) <= 1.5 else base
    t_tensor = _t_to_tensor(t, spectrum.device)
    if t_tensor.ndim == 0:
        filt = t_tensor * spectrum / (t_tensor.square() * spectrum + (1.0 - t_tensor).square() + float(eps))
    else:
        t_view = t_tensor.flatten().view(-1, 1, 1)
        filt = t_view * spectrum.view(1, *spectrum.shape[-2:]) / (
            t_view.square() * spectrum.view(1, *spectrum.shape[-2:]) + (1.0 - t_view).square() + float(eps)
        )
        filt = filt.unsqueeze(1)
    if normalize_mean:
        filt = normalize_filter_mean(filt, eps=eps)
    return filt


def _downsample(x: torch.Tensor, size: int) -> tuple[torch.Tensor, bool]:
    if x.ndim == 3:
        x_b = x.unsqueeze(0)
        squeezed = True
    elif x.ndim == 4:
        x_b = x
        squeezed = False
    else:
        raise ValueError(f"Expected x_t shape [C,H,W] or [B,C,H,W], got {tuple(x.shape)}")
    if int(size) > 0 and x_b.shape[-2:] != (int(size), int(size)):
        x_b = F.interpolate(x_b.to(torch.float32), size=(int(size), int(size)), mode="bilinear", align_corners=False)
    else:
        x_b = x_b.to(torch.float32)
    return x_b, squeezed


def wiener_clean_proxy(
    x_t: torch.Tensor,
    t: float | torch.Tensor,
    size: int = 64,
    beta: float = 2.0,
    f0: float = 0.03,
    eps: float = 1e-8,
    save_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Estimate a low-resolution clean image proxy from model-space ``x_t``."""

    x_small, squeezed = _downsample(x_t, size)
    height, width = x_small.shape[-2:]
    freq = radial_frequency_grid(height, width, device=x_small.device, dtype=torch.float32, normalize=True)
    filt = wiener_filter_xpred(t, freq, beta=beta, f0=f0, eps=eps, normalize_mean=False)
    proxy = fft_filter_2d(x_small, filt, norm="ortho")
    if save_dtype is not None:
        proxy = proxy.to(save_dtype)
    return proxy.squeeze(0) if squeezed else proxy

