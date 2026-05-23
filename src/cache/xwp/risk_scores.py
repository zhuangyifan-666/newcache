"""Risk-score utilities for the E6-D0 xWPCache-v2 offline diagnostic."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .frequency import natural_image_spectrum, radial_frequency_grid
from .perceptual_weight import perceptual_frequency_weight, phi_perceptual
from .wiener_proxy import wiener_clean_proxy


def _as_bchw(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim == 4:
        return x, False
    raise ValueError(f"Expected tensor shape [C,H,W] or [B,C,H,W], got {tuple(x.shape)}")


def _downsample(x: torch.Tensor, size: int) -> torch.Tensor:
    x_b, squeezed = _as_bchw(x)
    x_b = x_b.to(torch.float32)
    if int(size) > 0 and x_b.shape[-2:] != (int(size), int(size)):
        x_b = F.interpolate(x_b, size=(int(size), int(size)), mode="bilinear", align_corners=False)
    return x_b.squeeze(0) if squeezed else x_b


def _to_device_tensor(x: torch.Tensor, device: torch.device | None) -> torch.Tensor:
    return x.to(device=device, dtype=torch.float32) if device is not None else x.to(torch.float32)


def symmetric_relative_l1(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8, reduce: bool = True) -> torch.Tensor:
    """Symmetric relative L1: ``2||a-b||_1 / (||a||_1 + ||b||_1 + eps)``."""

    a_b, a_squeezed = _as_bchw(a)
    b_b, _ = _as_bchw(b)
    a_b = a_b.to(torch.float32)
    b_b = b_b.to(device=a_b.device, dtype=torch.float32)
    numerator = 2.0 * (a_b - b_b).abs().sum(dim=(1, 2, 3))
    denominator = a_b.abs().sum(dim=(1, 2, 3)) + b_b.abs().sum(dim=(1, 2, 3)) + float(eps)
    value = numerator / denominator
    if reduce:
        return value.mean()
    return value.squeeze(0) if a_squeezed and value.numel() == 1 else value


def ode_factor(t: float | torch.Tensor, h: float | torch.Tensor, epsilon_clip: float = 0.05):
    """ODE amplification factor ``abs(h) / max(1 - t, epsilon_clip)``."""

    if isinstance(t, torch.Tensor) or isinstance(h, torch.Tensor):
        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(float(t), dtype=torch.float32)
        h_tensor = h if isinstance(h, torch.Tensor) else torch.tensor(float(h), dtype=torch.float32, device=t_tensor.device)
        return h_tensor.to(torch.float32).abs() / (1.0 - t_tensor.to(torch.float32)).clamp_min(float(epsilon_clip))
    return abs(float(h)) / max(1.0 - float(t), float(epsilon_clip))


def posterior_uncertainty_xpred(
    t: float | torch.Tensor,
    size: int = 64,
    beta: float = 2.0,
    f0: float = 0.03,
    perceptual_weight: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Frequency-domain posterior clean-proxy uncertainty for x-prediction."""

    device = t.device if isinstance(t, torch.Tensor) else None
    freq = radial_frequency_grid(size, size, device=device, dtype=torch.float32, normalize=True)
    sx = natural_image_spectrum(freq, beta=beta, f0=f0, eps=eps)
    t_tensor = t.to(device=freq.device, dtype=torch.float32) if isinstance(t, torch.Tensor) else torch.tensor(float(t), device=freq.device)
    if t_tensor.ndim == 0:
        var = sx * (1.0 - t_tensor).square() / (t_tensor.square() * sx + (1.0 - t_tensor).square() + float(eps))
        weight = perceptual_frequency_weight(t_tensor, size=size, device=freq.device) if perceptual_weight else torch.ones_like(sx)
        return torch.sqrt((weight.square() * var).sum() / ((weight.square() * sx).sum() + float(eps)))
    values = []
    for item in t_tensor.flatten():
        values.append(posterior_uncertainty_xpred(item, size=size, beta=beta, f0=f0, perceptual_weight=perceptual_weight, eps=eps))
    return torch.stack(values)


def perceptual_snr_uncertainty(
    t: float | torch.Tensor,
    size: int = 64,
    beta: float = 2.0,
    f0: float = 0.03,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Alternative uncertainty ``1 / sqrt(SNR_P + eps)``."""

    device = t.device if isinstance(t, torch.Tensor) else None
    freq = radial_frequency_grid(size, size, device=device, dtype=torch.float32, normalize=True)
    sx = natural_image_spectrum(freq, beta=beta, f0=f0, eps=eps)
    t_tensor = t.to(device=freq.device, dtype=torch.float32) if isinstance(t, torch.Tensor) else torch.tensor(float(t), device=freq.device)
    if t_tensor.ndim == 0:
        weight = perceptual_frequency_weight(t_tensor, size=size, device=freq.device)
        signal = (weight.square() * t_tensor.square() * sx).sum()
        noise = (weight.square() * (1.0 - t_tensor).square()).sum()
        snr = signal / (noise + float(eps))
        return 1.0 / torch.sqrt(snr + float(eps))
    return torch.stack([perceptual_snr_uncertainty(item, size=size, beta=beta, f0=f0, eps=eps) for item in t_tensor.flatten()])


def _current_proxy(
    call: Mapping[str, Any],
    *,
    proxy_mode: str,
    size: int,
    beta: float,
    f0: float,
    eps: float,
    device: torch.device | None,
) -> torch.Tensor:
    t = float(call["t"])
    if proxy_mode == "raw":
        return _downsample(_to_device_tensor(call["x_t"], device), size)
    if proxy_mode == "wiener":
        return wiener_clean_proxy(_to_device_tensor(call["x_t"], device), t, size=size, beta=beta, f0=f0, eps=eps)
    if proxy_mode == "oracle_xhat":
        return _downsample(_to_device_tensor(call["xhat"], device), size)
    raise ValueError(f"Unknown proxy_mode: {proxy_mode}")


def _phi_or_identity(x: torch.Tensor, t: float, *, use_perceptual: bool, size: int) -> torch.Tensor:
    if use_perceptual:
        return phi_perceptual(x, t, size=size)
    return _downsample(x, size)


def scalar_window_risk(
    calls: Sequence[Mapping[str, Any]],
    anchor_xhat: torch.Tensor,
    proxy_mode: str = "wiener",
    use_perceptual: bool = True,
    use_ode: bool = True,
    use_uncertainty: bool = False,
    eta: float = 0.0,
    size: int = 64,
    beta: float = 2.0,
    f0: float = 0.03,
    epsilon_clip: float = 0.05,
    eps: float = 1e-8,
    device: torch.device | None = None,
    return_details: bool = False,
) -> float | Tuple[float, Dict[str, Any]]:
    """Accumulate scalar anchor-relative risk over a window of calls."""

    anchor = _to_device_tensor(anchor_xhat, device)
    score = 0.0
    uncertainty_sum = 0.0
    distances = []
    coeffs = []
    for call in calls:
        t = float(call["t"])
        h = float(call.get("effective_solver_coeff", call.get("h", 0.0)))
        coeff = float(ode_factor(t, h, epsilon_clip=epsilon_clip)) if use_ode else 1.0
        current = _current_proxy(call, proxy_mode=proxy_mode, size=size, beta=beta, f0=f0, eps=eps, device=device)
        z_anchor = _phi_or_identity(anchor, t, use_perceptual=use_perceptual, size=size)
        z_current = _phi_or_identity(current, t, use_perceptual=use_perceptual, size=size)
        dist = float(symmetric_relative_l1(z_anchor, z_current, eps=eps).detach().cpu())
        score += coeff * dist
        distances.append(dist)
        coeffs.append(coeff)
        if use_uncertainty or eta != 0.0:
            uncertainty_sum += float(
                posterior_uncertainty_xpred(t, size=size, beta=beta, f0=f0, perceptual_weight=use_perceptual, eps=eps)
                .detach()
                .cpu()
            )
    score += float(eta) * uncertainty_sum
    if not return_details:
        return float(score)
    return float(score), {
        "distances": distances,
        "coefficients": coeffs,
        "uncertainty_sum": uncertainty_sum,
        "num_calls": len(calls),
    }


def vector_window_risk(
    calls: Sequence[Mapping[str, Any]],
    anchor_xhat: torch.Tensor,
    proxy_mode: str = "wiener",
    use_perceptual: bool = True,
    use_ode: bool = True,
    use_uncertainty: bool = False,
    eta: float = 0.0,
    size: int = 64,
    beta: float = 2.0,
    f0: float = 0.03,
    epsilon_clip: float = 0.05,
    eps: float = 1e-8,
    device: torch.device | None = None,
) -> Dict[str, float]:
    """Vector accumulated solver risk from the xWPCache-v2 formula."""

    anchor = _to_device_tensor(anchor_xhat, device)
    residual_sum = None
    denominator = 0.0
    uncertainty_sum = 0.0
    for call in calls:
        t = float(call["t"])
        h = float(call.get("effective_solver_coeff", call.get("h", 0.0)))
        coeff = float(ode_factor(t, h, epsilon_clip=epsilon_clip)) if use_ode else 1.0
        current = _current_proxy(call, proxy_mode=proxy_mode, size=size, beta=beta, f0=f0, eps=eps, device=device)
        z_anchor = _phi_or_identity(anchor, t, use_perceptual=use_perceptual, size=size)
        z_current = _phi_or_identity(current, t, use_perceptual=use_perceptual, size=size)
        residual = coeff * (z_anchor - z_current)
        residual_sum = residual if residual_sum is None else residual_sum + residual
        denominator += coeff * float((z_anchor.abs().sum() + z_current.abs().sum()).detach().cpu())
        if use_uncertainty or eta != 0.0:
            uncertainty_sum += float(
                posterior_uncertainty_xpred(t, size=size, beta=beta, f0=f0, perceptual_weight=use_perceptual, eps=eps)
                .detach()
                .cpu()
            )

    if residual_sum is None:
        numerator = 0.0
        risk = 0.0
    else:
        numerator = 2.0 * float(residual_sum.abs().sum().detach().cpu())
        risk = numerator / (denominator + float(eps))
    risk += float(eta) * uncertainty_sum
    return {
        "risk": float(risk),
        "numerator": float(numerator),
        "denominator": float(denominator),
        "uncertainty_sum": float(uncertainty_sum),
        "num_calls": float(len(calls)),
    }
