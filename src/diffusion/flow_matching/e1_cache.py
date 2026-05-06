from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional, Set, Tuple

import torch
import torch.nn as nn


def unwrap_runtime_module(module: nn.Module) -> nn.Module:
    current = module
    while True:
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "_orig_mod"):
            current = current._orig_mod
            continue
        return current


def _reshape_tokens_to_map(tokens: torch.Tensor) -> Tuple[torch.Tensor, int]:
    batch, num_tokens, channels = tokens.shape
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Expected a square token grid, but got {num_tokens} tokens.")
    feature_map = tokens.view(batch, side, side, channels).permute(0, 3, 1, 2).contiguous()
    return feature_map, side


def build_sea_filter(
    height: int,
    width: int,
    t: torch.Tensor,
    beta: float,
    eps: float,
) -> torch.Tensor:
    freq_y = torch.fft.fftfreq(height, device=t.device, dtype=torch.float32).view(height, 1)
    freq_x = torch.fft.fftfreq(width, device=t.device, dtype=torch.float32).view(1, width)
    radius_sq = (freq_y.square() + freq_x.square()).clamp_min(eps)
    signal_prior = radius_sq.pow(-0.5 * beta)

    signal_coeff = t.to(torch.float32).clamp_min(eps).view(-1, 1, 1)
    noise_coeff = (1.0 - t.to(torch.float32)).clamp_min(eps).view(-1, 1, 1)
    signal_power = signal_coeff.square() * signal_prior
    noise_power = noise_coeff.square()

    filt = signal_power / (signal_power + noise_power + eps)
    filt = filt / filt.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return filt


def apply_sea_filter(proxy_tokens: torch.Tensor, t: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
    feature_map, side = _reshape_tokens_to_map(proxy_tokens)
    spectrum = torch.fft.fftn(feature_map.to(torch.float32), dim=(-2, -1))
    filt = build_sea_filter(side, side, t, beta=beta, eps=eps).unsqueeze(1)
    filtered = torch.fft.ifftn(spectrum * filt, dim=(-2, -1)).real
    filtered = filtered.permute(0, 2, 3, 1).reshape_as(proxy_tokens)
    return filtered.to(proxy_tokens.dtype)


def relative_l1_distance(current: torch.Tensor, previous: torch.Tensor, eps: float) -> float:
    numerator = (current - previous).abs().mean(dim=(1, 2))
    denominator = previous.abs().mean(dim=(1, 2)).clamp_min(eps)
    return float((numerator / denominator).mean().item())


def extract_jit_modulated_proxy(model: nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    required_attrs = ("t_embedder", "y_embedder", "x_embedder", "pos_embed", "blocks")
    if not all(hasattr(model, attr) for attr in required_attrs):
        raise TypeError("E1 cache currently supports JiT-like denoisers only.")

    t_emb = model.t_embedder(t)
    y_emb = model.y_embedder(y)
    c = t_emb + y_emb

    tokens = model.x_embedder(x)
    tokens = tokens + model.pos_embed.to(device=tokens.device, dtype=tokens.dtype)

    first_block = model.blocks[0]
    shift_msa, scale_msa, _, _, _, _ = first_block.adaLN_modulation(c).chunk(6, dim=-1)
    proxy = first_block.norm1(tokens)
    proxy = proxy * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    return proxy


@dataclass
class CacheStats:
    queries: int = 0
    hits: int = 0
    refreshes: int = 0
    forced_refreshes: int = 0
    total_rel_l1: float = 0.0
    max_consecutive_hits: int = 0
    denoiser_time_sec: float = 0.0

    @property
    def refresh_ratio(self) -> float:
        if self.queries <= 0:
            return 0.0
        return self.refreshes / self.queries

    @property
    def hit_rate(self) -> float:
        if self.queries <= 0:
            return 0.0
        return self.hits / self.queries

    @property
    def avg_rel_l1(self) -> float:
        if self.queries <= 0:
            return 0.0
        return self.total_rel_l1 / self.queries


@dataclass
class _CacheState:
    cached_output: Optional[torch.Tensor] = None
    previous_proxy: Optional[torch.Tensor] = None
    accumulated_distance: float = 0.0
    consecutive_hits: int = 0
    uniform_refresh_calls: Optional[Set[int]] = None


class BaseE1CacheController:
    def __init__(self) -> None:
        self.state = _CacheState()
        self.stats = CacheStats()

    def reset(self) -> None:
        self.state = _CacheState()
        self.stats = CacheStats()

    def start_sample(self, total_calls: int) -> None:
        del total_calls
        self.state = _CacheState()

    def _record(self, *, hit: bool, forced_refresh: bool, rel_l1: float = 0.0) -> None:
        self.stats.queries += 1
        self.stats.total_rel_l1 += float(rel_l1)
        self.stats.max_consecutive_hits = max(self.stats.max_consecutive_hits, self.state.consecutive_hits)
        if hit:
            self.stats.hits += 1
            return
        self.stats.refreshes += 1
        if forced_refresh:
            self.stats.forced_refreshes += 1

    def _run_denoiser(self, net: nn.Module, cfg_x: torch.Tensor, cfg_t: torch.Tensor, cfg_condition: torch.Tensor):
        if cfg_x.is_cuda:
            torch.cuda.synchronize(cfg_x.device)
        start = time.perf_counter()
        out = net(cfg_x, cfg_t, cfg_condition)
        if cfg_x.is_cuda:
            torch.cuda.synchronize(cfg_x.device)
        self.stats.denoiser_time_sec += time.perf_counter() - start
        return out

    def predict_cfg_output(
        self,
        net: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        cfg_condition: torch.Tensor,
        call_index: int,
        total_calls: int,
    ):
        raise NotImplementedError


class AlwaysRefreshController(BaseE1CacheController):
    def predict_cfg_output(
        self,
        net: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        cfg_condition: torch.Tensor,
        call_index: int,
        total_calls: int,
    ):
        del call_index, total_calls
        cfg_x = torch.cat([x, x], dim=0)
        cfg_t = t.repeat(2)
        cfg_output = self._run_denoiser(net, cfg_x, cfg_t, cfg_condition)
        self.state.cached_output = cfg_output.detach()
        self.state.consecutive_hits = 0
        self._record(hit=False, forced_refresh=True)
        return cfg_output, cfg_x, cfg_t


class UniformCacheController(BaseE1CacheController):
    def __init__(self, target_rr: float) -> None:
        super().__init__()
        if target_rr <= 0.0 or target_rr > 1.0:
            raise ValueError("target_rr must be in (0, 1].")
        self.target_rr = float(target_rr)

    @staticmethod
    def _build_schedule(total_calls: int, target_rr: float) -> Set[int]:
        refresh_count = int(round(total_calls * target_rr))
        refresh_count = max(1, min(total_calls, refresh_count))
        if refresh_count == total_calls:
            return set(range(total_calls))
        if refresh_count == 1:
            return {0}
        return {
            int(round(idx * (total_calls - 1) / (refresh_count - 1)))
            for idx in range(refresh_count)
        }

    def start_sample(self, total_calls: int) -> None:
        super().start_sample(total_calls)
        self.state.uniform_refresh_calls = self._build_schedule(total_calls, self.target_rr)

    def predict_cfg_output(
        self,
        net: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        cfg_condition: torch.Tensor,
        call_index: int,
        total_calls: int,
    ):
        del total_calls
        cfg_x = torch.cat([x, x], dim=0)
        cfg_t = t.repeat(2)
        schedule = self.state.uniform_refresh_calls or {0}
        forced_refresh = self.state.cached_output is None or call_index in schedule

        if forced_refresh:
            cfg_output = self._run_denoiser(net, cfg_x, cfg_t, cfg_condition)
            self.state.cached_output = cfg_output.detach()
            self.state.consecutive_hits = 0
            self._record(hit=False, forced_refresh=True)
            return cfg_output, cfg_x, cfg_t

        self.state.consecutive_hits += 1
        self._record(hit=True, forced_refresh=False)
        return self.state.cached_output, cfg_x, cfg_t


class OnlineInputCacheController(BaseE1CacheController):
    def __init__(
        self,
        metric: str,
        delta: float,
        warmup_calls: int = 5,
        max_skip_calls: int = 4,
        sea_filter_beta: float = 2.0,
        sea_filter_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if metric not in {"raw", "sea"}:
            raise ValueError("metric must be either 'raw' or 'sea'.")
        self.metric = metric
        self.delta = float(delta)
        self.warmup_calls = int(warmup_calls)
        self.max_skip_calls = int(max_skip_calls)
        self.sea_filter_beta = float(sea_filter_beta)
        self.sea_filter_eps = float(sea_filter_eps)

    def _proxy(self, net: nn.Module, cfg_x: torch.Tensor, cfg_t: torch.Tensor, cfg_condition: torch.Tensor) -> torch.Tensor:
        runtime_model = unwrap_runtime_module(net)
        proxy = extract_jit_modulated_proxy(runtime_model, cfg_x, cfg_t, cfg_condition)
        if self.metric == "sea":
            proxy = apply_sea_filter(proxy, cfg_t, beta=self.sea_filter_beta, eps=self.sea_filter_eps)
        return proxy

    def predict_cfg_output(
        self,
        net: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        cfg_condition: torch.Tensor,
        call_index: int,
        total_calls: int,
    ):
        cfg_x = torch.cat([x, x], dim=0)
        cfg_t = t.repeat(2)

        proxy = self._proxy(net, cfg_x, cfg_t, cfg_condition)
        forced_refresh = (
            self.state.cached_output is None
            or call_index < self.warmup_calls
            or call_index == total_calls - 1
        )

        rel_l1 = 0.0
        if self.state.previous_proxy is not None:
            rel_l1 = relative_l1_distance(proxy, self.state.previous_proxy, eps=self.sea_filter_eps)

        if not forced_refresh:
            self.state.accumulated_distance += rel_l1

        should_refresh = (
            forced_refresh
            or self.state.previous_proxy is None
            or self.state.accumulated_distance >= self.delta
            or self.state.consecutive_hits >= self.max_skip_calls
        )

        if should_refresh:
            cfg_output = self._run_denoiser(net, cfg_x, cfg_t, cfg_condition)
            self.state.cached_output = cfg_output.detach()
            self.state.previous_proxy = proxy.detach()
            self.state.accumulated_distance = 0.0
            self.state.consecutive_hits = 0
            self._record(hit=False, forced_refresh=forced_refresh, rel_l1=rel_l1)
            return cfg_output, cfg_x, cfg_t

        self.state.consecutive_hits += 1
        self.state.previous_proxy = proxy.detach()
        self._record(hit=True, forced_refresh=False, rel_l1=rel_l1)
        return self.state.cached_output, cfg_x, cfg_t
