"""xWPCache-v2 offline diagnostic math utilities."""

from .frequency import (
    fft_filter_2d,
    natural_image_spectrum,
    normalize_filter_mean,
    radial_frequency_grid,
)
from .perceptual_weight import perceptual_frequency_weight, phi_perceptual, q_noise_gate
from .risk_scores import (
    ode_factor,
    perceptual_snr_uncertainty,
    posterior_uncertainty_xpred,
    scalar_window_risk,
    symmetric_relative_l1,
    vector_window_risk,
)
from .wiener_proxy import wiener_clean_proxy, wiener_filter_xpred

__all__ = [
    "fft_filter_2d",
    "natural_image_spectrum",
    "normalize_filter_mean",
    "radial_frequency_grid",
    "ode_factor",
    "perceptual_frequency_weight",
    "perceptual_snr_uncertainty",
    "phi_perceptual",
    "posterior_uncertainty_xpred",
    "q_noise_gate",
    "scalar_window_risk",
    "symmetric_relative_l1",
    "vector_window_risk",
    "wiener_clean_proxy",
    "wiener_filter_xpred",
]

