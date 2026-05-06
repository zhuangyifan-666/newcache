#!/usr/bin/env python3
"""Single-GPU E1 online cache baselines for PixelGen.

This script implements the Phase-1 E1 pilot:

- Full inference reference
- Uniform matched-RR output cache
- RawInput online output cache
- SEAInput online output cache
- Paired final-image PSNR / SSIM / optional LPIPS against the full reference

Recommended pilot:

conda run -n pixelgen python scripts/01_e1_online_cache.py \
  --device cuda:0 \
  --num-samples 64 \
  --batch-size 1 \
  --target-rrs 0.30,0.50 \
  --calibrate-online \
  --calib-samples 8 \
  --no-autocast \
  --run-id e1_pilot_64_fp32
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion.flow_matching.e1_cache import (  # noqa: E402
    AlwaysRefreshController,
    BaseE1CacheController,
    OnlineInputCacheController,
    UniformCacheController,
)
from src.models.autoencoder.base import fp2uint8  # noqa: E402


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_method_list(value: str) -> List[str]:
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"uniform", "raw", "sea"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")
    return methods


def import_symbol(path: str) -> Any:
    module_name, symbol_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def maybe_resolve_symbol(value: str) -> Any:
    if "/" in value or value.startswith("~"):
        return value
    if "." not in value:
        return value
    try:
        symbol = import_symbol(value)
    except Exception:
        return value
    if isinstance(symbol, type):
        return symbol()
    return symbol


def instantiate_from_config(config: Any) -> Any:
    if isinstance(config, (DictConfig, ListConfig)):
        config = OmegaConf.to_container(config, resolve=True)

    if isinstance(config, dict) and "class_path" in config:
        cls = import_symbol(str(config["class_path"]))
        init_args = config.get("init_args", {}) or {}
        kwargs = {key: instantiate_from_config(value) for key, value in init_args.items()}
        return cls(**kwargs)

    if isinstance(config, dict):
        return {key: instantiate_from_config(value) for key, value in config.items()}

    if isinstance(config, list):
        return [instantiate_from_config(value) for value in config]

    if isinstance(config, str):
        return maybe_resolve_symbol(config)

    return config


def load_denoiser_weights(
    denoiser: nn.Module,
    ckpt_path: Path,
    prefixes: Sequence[str],
) -> Dict[str, Any]:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = denoiser.state_dict()
    new_state = OrderedDict()
    loaded = 0
    missing: List[str] = []
    mismatched: List[str] = []
    used_prefixes = set()

    for name, tensor in model_state.items():
        matched_key = None
        for prefix in prefixes:
            candidate = f"{prefix}{name}"
            if candidate in state_dict:
                matched_key = candidate
                break

        if matched_key is None:
            new_state[name] = tensor
            missing.append(name)
            continue

        value = state_dict[matched_key]
        if tuple(value.shape) != tuple(tensor.shape):
            new_state[name] = tensor
            mismatched.append(name)
            continue

        new_state[name] = value
        loaded += 1
        used_prefixes.add(matched_key[: -len(name)])

    if loaded == 0:
        raise RuntimeError(
            f"No denoiser weights were loaded from {ckpt_path}. "
            f"Tried prefixes: {list(prefixes)}"
        )

    denoiser.load_state_dict(new_state, strict=False)
    return {
        "loaded": loaded,
        "total": len(model_state),
        "missing": len(missing),
        "mismatched": len(mismatched),
        "used_prefixes": sorted(used_prefixes),
        "missing_examples": missing[:10],
        "mismatched_examples": mismatched[:10],
    }


def configure_runtime(
    seed: int,
    deterministic_algorithms: bool,
    allow_tf32: bool,
    force_math_sdpa: bool,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if force_math_sdpa and torch.cuda.is_available():
        for name, enabled in [
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_cudnn_sdp", False),
            ("enable_math_sdp", True),
        ]:
            fn = getattr(torch.backends.cuda, name, None)
            if fn is not None:
                fn(enabled)


def class_seed_pairs(args: argparse.Namespace, num_classes: int) -> List[Tuple[int, int]]:
    classes = parse_int_list(args.classes)
    seeds = parse_int_list(args.seeds)

    if classes is None:
        classes = [idx % num_classes for idx in range(args.num_samples)]
    if seeds is None:
        seeds = list(range(args.seed_start, args.seed_start + len(classes)))

    if len(classes) != len(seeds):
        raise ValueError("--classes and --seeds must have the same length")
    return list(zip(classes, seeds))


def write_pairs_csv(pairs: Sequence[Tuple[int, int]], output_dir: Path) -> None:
    with open(output_dir / "class_seed_pairs.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["index", "class_id", "seed"])
        writer.writeheader()
        for idx, (class_id, seed) in enumerate(pairs):
            writer.writerow({"index": idx, "class_id": class_id, "seed": seed})


def make_noise_batch(
    pairs: Sequence[Tuple[int, int]],
    latent_shape: Sequence[int],
    device: torch.device,
    noise_scale: float,
) -> torch.Tensor:
    samples = []
    for _, seed in pairs:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        samples.append(noise_scale * torch.randn(tuple(latent_shape), generator=generator, dtype=torch.float32))
    return torch.stack(samples, dim=0).to(device)


def image_to_uint8_hwc(image: torch.Tensor) -> Any:
    image_u8 = fp2uint8(image.detach().cpu().unsqueeze(0).clone())[0]
    return image_u8.permute(1, 2, 0).numpy()


def save_tensor_image(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_to_uint8_hwc(image)).save(path)


def save_image_grid(images: torch.Tensor, path: Path, resize: int, columns: int) -> None:
    if images.numel() == 0:
        return
    images = images.detach().cpu().float().clamp(-1, 1)
    if resize > 0 and (images.shape[-1] != resize or images.shape[-2] != resize):
        images = F.interpolate(images, size=(resize, resize), mode="bilinear", align_corners=False)

    images_u8 = fp2uint8(images.clone())
    n, c, h, w = images_u8.shape
    columns = max(1, min(columns, n))
    rows = int(math.ceil(n / columns))
    grid = torch.full((c, rows * h, columns * w), 255, dtype=torch.uint8)
    for idx in range(n):
        row = idx // columns
        col = idx % columns
        grid[:, row * h : (row + 1) * h, col * w : (col + 1) * w] = images_u8[idx]

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid.permute(1, 2, 0).numpy()).save(path)


def total_denoiser_opportunities(sampler: nn.Module) -> int:
    if getattr(sampler, "exact_henu", False):
        return int(2 * sampler.num_steps - 1)
    return int(sampler.num_steps)


@torch.inference_mode()
def heun_jit_e1_sampling(
    net: nn.Module,
    sampler: nn.Module,
    noise: torch.Tensor,
    condition: torch.Tensor,
    uncondition: torch.Tensor,
    controller: BaseE1CacheController,
) -> torch.Tensor:
    batch_size = noise.shape[0]
    steps = sampler.timesteps.to(noise.device)
    cfg_condition = torch.cat([uncondition, condition], dim=0)
    total_calls = total_denoiser_opportunities(sampler)
    controller.start_sample(total_calls)

    x = noise
    v_hat, s_hat = 0.0, 0.0
    call_index = 0

    for i, (t_cur_scalar, t_next_scalar) in enumerate(zip(steps[:-1], steps[1:])):
        dt = t_next_scalar - t_cur_scalar
        t_cur = t_cur_scalar.repeat(batch_size).to(noise.device, noise.dtype)
        sigma = sampler.scheduler.sigma(t_cur)
        alpha_over_dalpha = 1 / sampler.scheduler.dalpha_over_alpha(t_cur)
        dsigma_mul_sigma = sampler.scheduler.dsigma_mul_sigma(t_cur)
        t_hat = t_next_scalar.repeat(batch_size).to(noise.device, noise.dtype)
        sigma_hat = sampler.scheduler.sigma(t_hat)
        alpha_over_dalpha_hat = 1 / sampler.scheduler.dalpha_over_alpha(t_hat)
        dsigma_mul_sigma_hat = sampler.scheduler.dsigma_mul_sigma(t_hat)
        w = sampler.w_scheduler.w(t_cur) if sampler.w_scheduler else 0.0

        if i == 0 or sampler.exact_henu:
            cfg_out, cfg_x, cfg_t_cur = controller.predict_cfg_output(
                net=net,
                x=x,
                t=t_cur,
                cfg_condition=cfg_condition,
                call_index=call_index,
                total_calls=total_calls,
            )
            call_index += 1
            out = (cfg_out - cfg_x) / (1.0 - cfg_t_cur.view(-1, 1, 1, 1)).clamp_min(sampler.t_eps)
            if t_cur[0] > sampler.guidance_interval_min and t_cur[0] <= sampler.guidance_interval_max:
                out = sampler.guidance_fn(out, sampler.guidance)
            else:
                out = sampler.guidance_fn(out, 1.0)
            v = out
            s = ((alpha_over_dalpha) * v - x) / (sigma**2 - (alpha_over_dalpha) * dsigma_mul_sigma)
        else:
            v = v_hat
            s = s_hat

        x_hat = sampler.step_fn(x, v, dt, s=s, w=w)

        if i < sampler.num_steps - 1:
            cfg_out_hat, cfg_x_hat, cfg_t_hat = controller.predict_cfg_output(
                net=net,
                x=x_hat,
                t=t_hat,
                cfg_condition=cfg_condition,
                call_index=call_index,
                total_calls=total_calls,
            )
            call_index += 1
            out = (cfg_out_hat - cfg_x_hat) / (1.0 - cfg_t_hat.view(-1, 1, 1, 1)).clamp_min(sampler.t_eps)
            if t_cur[0] > sampler.guidance_interval_min and t_cur[0] <= sampler.guidance_interval_max:
                out = sampler.guidance_fn(out, sampler.guidance)
            else:
                out = sampler.guidance_fn(out, 1.0)

            v_hat = out
            s_hat = ((alpha_over_dalpha_hat) * v_hat - x_hat) / (
                sigma_hat**2 - (alpha_over_dalpha_hat) * dsigma_mul_sigma_hat
            )
            v = (v + v_hat) / 2
            s = (s + s_hat) / 2
            x = sampler.step_fn(x, v, dt, s=s, w=w)
        else:
            x = sampler.last_step_fn(x, v, dt, s=s, w=w)

    if call_index != total_calls:
        raise RuntimeError(f"Expected {total_calls} denoiser opportunities, got {call_index}.")
    return x


def gaussian_window(channels: int, device: torch.device, dtype: torch.dtype, size: int = 11, sigma: float = 1.5):
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    kernel_1d = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, size, size)
    return kernel_2d.expand(channels, 1, size, size).to(dtype=dtype)


def ssim_per_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = ((x.float().clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
    y = ((y.float().clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
    _, channels, _, _ = x.shape
    window = gaussian_window(channels, x.device, x.dtype)
    padding = window.shape[-1] // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ).clamp_min(1e-12)
    return ssim_map.mean(dim=(1, 2, 3))


def load_lpips_model(args: argparse.Namespace, device: torch.device):
    if args.skip_lpips:
        return None, "skipped"
    try:
        import lpips  # type: ignore

        model = lpips.LPIPS(net=args.lpips_net).to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model, ""
    except Exception as exc:  # pragma: no cover - depends on optional local package/data.
        return None, str(exc)


@torch.inference_mode()
def compute_batch_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    device: torch.device,
    lpips_model: Optional[nn.Module],
) -> Dict[str, List[Optional[float]]]:
    ref = reference.to(device=device, dtype=torch.float32)
    cand = candidate.to(device=device, dtype=torch.float32)
    diff = cand - ref
    mse = diff.square().mean(dim=(1, 2, 3))
    psnr = torch.where(
        mse == 0,
        torch.full_like(mse, float("inf")),
        20.0 * torch.log10(torch.tensor(2.0, device=device) / torch.sqrt(mse.clamp_min(1e-30))),
    )
    ssim = ssim_per_image(cand, ref)

    if lpips_model is not None:
        lpips_values = lpips_model(cand.clamp(-1, 1), ref.clamp(-1, 1)).view(-1).detach().float().cpu().tolist()
    else:
        lpips_values = [None for _ in range(candidate.shape[0])]

    return {
        "psnr_db": [float(value) for value in psnr.detach().cpu().tolist()],
        "ssim": [float(value) for value in ssim.detach().cpu().tolist()],
        "lpips": lpips_values,
    }


def summarize_metric(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": sum(finite) / len(finite),
        "min": min(finite),
        "max": max(finite),
    }


def stats_to_summary(controller: BaseE1CacheController) -> Dict[str, Any]:
    stats = controller.stats
    return {
        "cache_queries": stats.queries,
        "cache_hits": stats.hits,
        "cache_refreshes": stats.refreshes,
        "cache_forced_refreshes": stats.forced_refreshes,
        "actual_rr": stats.refresh_ratio,
        "cache_hit_rate": stats.hit_rate,
        "avg_rel_l1": stats.avg_rel_l1,
        "max_consecutive_hits": stats.max_consecutive_hits,
        "denoiser_time_sec": stats.denoiser_time_sec,
    }


def run_sampling_only(
    pairs: Sequence[Tuple[int, int]],
    latent_shape: Sequence[int],
    denoiser: nn.Module,
    conditioner: nn.Module,
    sampler: nn.Module,
    controller: BaseE1CacheController,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    controller.reset()
    autocast_dtype = torch.bfloat16 if args.autocast_dtype == "bf16" else torch.float16
    start = time.perf_counter()
    for batch_start in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[batch_start : batch_start + args.batch_size]
        labels = [pair[0] for pair in batch_pairs]
        noise = make_noise_batch(batch_pairs, latent_shape, device, args.noise_scale)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=not args.no_autocast):
            condition, uncondition = conditioner(labels, {})
            _ = heun_jit_e1_sampling(denoiser, sampler, noise, condition, uncondition, controller)
    elapsed = time.perf_counter() - start
    out = stats_to_summary(controller)
    out["elapsed_sec"] = elapsed
    return out


def run_method(
    name: str,
    pairs: Sequence[Tuple[int, int]],
    latent_shape: Sequence[int],
    denoiser: nn.Module,
    vae: nn.Module,
    conditioner: nn.Module,
    sampler: nn.Module,
    controller: BaseE1CacheController,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
    reference_images: Optional[torch.Tensor],
    lpips_model: Optional[nn.Module],
    return_images: bool,
) -> Dict[str, Any]:
    controller.reset()
    method_dir = output_dir / name
    image_dir = method_dir / "final_images"
    method_dir.mkdir(parents=True, exist_ok=True)
    if args.save_image_count > 0:
        image_dir.mkdir(parents=True, exist_ok=True)

    autocast_dtype = torch.bfloat16 if args.autocast_dtype == "bf16" else torch.float16
    final_images: List[torch.Tensor] = []
    metric_rows: List[Dict[str, Any]] = []
    preview_images: List[torch.Tensor] = []
    preview_image_count = 0
    start = time.perf_counter()

    print(f"[E1] Running {name} on {len(pairs)} samples", flush=True)
    for batch_start in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[batch_start : batch_start + args.batch_size]
        labels = [pair[0] for pair in batch_pairs]
        noise = make_noise_batch(batch_pairs, latent_shape, device, args.noise_scale)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=not args.no_autocast):
            condition, uncondition = conditioner(labels, {})
            samples = heun_jit_e1_sampling(denoiser, sampler, noise, condition, uncondition, controller)
            decoded = vae.decode(samples)

        decoded_cpu = decoded.detach().cpu().float()
        if return_images:
            final_images.append(decoded_cpu)

        if reference_images is not None:
            ref_cpu = reference_images[batch_start : batch_start + decoded_cpu.shape[0]]
            batch_metrics = compute_batch_metrics(ref_cpu, decoded_cpu, device=device, lpips_model=lpips_model)
            for local_idx, (class_id, seed) in enumerate(batch_pairs):
                metric_rows.append(
                    {
                        "index": batch_start + local_idx,
                        "class_id": class_id,
                        "seed": seed,
                        "psnr_db": batch_metrics["psnr_db"][local_idx],
                        "ssim": batch_metrics["ssim"][local_idx],
                        "lpips": batch_metrics["lpips"][local_idx],
                    }
                )

        remaining_to_save = args.save_image_count - preview_image_count
        if remaining_to_save > 0:
            take = min(remaining_to_save, decoded_cpu.shape[0])
            preview_images.append(decoded_cpu[:take])
            preview_image_count += take
            for local_idx, (class_id, seed) in enumerate(batch_pairs[:take]):
                save_tensor_image(decoded_cpu[local_idx], image_dir / f"class{class_id:04d}_seed{seed}.png")

    elapsed = time.perf_counter() - start
    if preview_images:
        save_image_grid(
            torch.cat(preview_images, dim=0),
            method_dir / "preview_grid.png",
            resize=args.preview_size,
            columns=args.preview_columns,
        )

    per_sample_csv = None
    metric_summary: Dict[str, Any] = {}
    if metric_rows:
        per_sample_csv = method_dir / "per_sample_metrics.csv"
        with open(per_sample_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)
        metric_summary = {
            "psnr_db": summarize_metric(row["psnr_db"] for row in metric_rows),
            "ssim": summarize_metric(row["ssim"] for row in metric_rows),
            "lpips": summarize_metric(row["lpips"] for row in metric_rows),
        }

    final_tensor = torch.cat(final_images, dim=0) if final_images else None
    summary = {
        "name": name,
        "num_samples": len(pairs),
        "elapsed_sec": elapsed,
        "per_sample_csv": str(per_sample_csv) if per_sample_csv else None,
        "method_dir": str(method_dir),
        "metrics": metric_summary,
        **stats_to_summary(controller),
    }
    if final_tensor is not None:
        summary["final_tensor"] = final_tensor
    return summary


def format_token(value: float) -> str:
    return f"{value:.4g}".replace(".", "p").replace("-", "m")


def choose_delta(
    calibration_rows: Sequence[Dict[str, Any]],
    target_rr: float,
) -> Dict[str, Any]:
    if not calibration_rows:
        raise ValueError("No calibration rows available.")
    return min(
        calibration_rows,
        key=lambda row: (abs(float(row["actual_rr"]) - target_rr), -float(row["actual_rr"])),
    )


def write_method_summary_csv(methods: Sequence[Dict[str, Any]], path: Path) -> None:
    fields = [
        "name",
        "kind",
        "target_rr",
        "delta",
        "num_samples",
        "elapsed_sec",
        "cache_queries",
        "cache_hits",
        "cache_refreshes",
        "cache_forced_refreshes",
        "actual_rr",
        "cache_hit_rate",
        "denoiser_time_sec",
        "max_consecutive_hits",
        "psnr_mean",
        "psnr_min",
        "ssim_mean",
        "ssim_min",
        "lpips_mean",
        "lpips_max",
        "per_sample_csv",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for item in methods:
            metrics = item.get("metrics", {}) or {}
            psnr = metrics.get("psnr_db", {}) or {}
            ssim = metrics.get("ssim", {}) or {}
            lpips_metric = metrics.get("lpips", {}) or {}
            writer.writerow(
                {
                    "name": item.get("name"),
                    "kind": item.get("kind"),
                    "target_rr": item.get("target_rr"),
                    "delta": item.get("delta"),
                    "num_samples": item.get("num_samples"),
                    "elapsed_sec": item.get("elapsed_sec"),
                    "cache_queries": item.get("cache_queries"),
                    "cache_hits": item.get("cache_hits"),
                    "cache_refreshes": item.get("cache_refreshes"),
                    "cache_forced_refreshes": item.get("cache_forced_refreshes"),
                    "actual_rr": item.get("actual_rr"),
                    "cache_hit_rate": item.get("cache_hit_rate"),
                    "denoiser_time_sec": item.get("denoiser_time_sec"),
                    "max_consecutive_hits": item.get("max_consecutive_hits"),
                    "psnr_mean": psnr.get("mean"),
                    "psnr_min": psnr.get("min"),
                    "ssim_mean": ssim.get("mean"),
                    "ssim_min": ssim.get("min"),
                    "lpips_mean": lpips_metric.get("mean"),
                    "lpips_max": lpips_metric.get("max"),
                    "per_sample_csv": item.get("per_sample_csv"),
                }
            )


def json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return "<tensor omitted>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs_c2i/PixelGen_XL_without_CFG.yaml")
    parser.add_argument("--ckpt", default="ckpts/PixelGen_XL_80ep.ckpt")
    parser.add_argument("--output-dir", default="outputs/e1_online_cache")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--classes", default=None, help="Comma-separated class ids. Defaults to 0..num_samples-1.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to seed_start..")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--methods", default="uniform,raw,sea")
    parser.add_argument("--target-rrs", default="0.30,0.50")
    parser.add_argument("--online-deltas", default="0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.50,0.80")
    parser.set_defaults(calibrate_online=False)
    parser.add_argument("--calibrate-online", dest="calibrate_online", action="store_true")
    parser.add_argument("--no-calibrate-online", dest="calibrate_online", action="store_false")
    parser.add_argument("--calib-samples", type=int, default=8)
    parser.add_argument("--warmup-calls", type=int, default=5)
    parser.add_argument("--max-skip-calls", type=int, default=4)
    parser.add_argument("--sea-filter-beta", type=float, default=2.0)
    parser.add_argument("--sea-filter-eps", type=float, default=1e-6)
    parser.add_argument("--autocast-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--allow-fused-sdpa", action="store_true")
    parser.add_argument("--global-seed", type=int, default=2026)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--save-image-count", type=int, default=16)
    parser.add_argument("--preview-size", type=int, default=128)
    parser.add_argument("--preview-columns", type=int, default=4)
    parser.add_argument(
        "--weight-prefixes",
        default="ema_denoiser.,denoiser.,",
        help="Comma-separated checkpoint prefixes to try in order.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. E1 requires a GPU-visible shell.")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    configure_runtime(
        args.global_seed,
        args.deterministic_algorithms,
        args.allow_tf32,
        force_math_sdpa=not args.allow_fused_sdpa,
    )

    device = torch.device(args.device)
    config_path = (PROJECT_ROOT / args.config).resolve()
    ckpt_path = (PROJECT_ROOT / args.ckpt).resolve()
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = (PROJECT_ROOT / args.output_dir / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(config_path)
    num_classes = int(config.model.denoiser.init_args.num_classes)
    latent_shape = config.data.pred_dataset.init_args.latent_shape
    pairs = class_seed_pairs(args, num_classes=num_classes)
    write_pairs_csv(pairs, output_dir)

    vae = instantiate_from_config(config.model.vae).to(device).eval()
    denoiser = instantiate_from_config(config.model.denoiser).to(device).eval()
    conditioner = instantiate_from_config(config.model.conditioner).to(device).eval()
    sampler = instantiate_from_config(config.model.diffusion_sampler).to(device).eval()

    prefixes = [item for item in args.weight_prefixes.split(",") if item != ""]
    if args.weight_prefixes.endswith(","):
        prefixes.append("")
    load_info = load_denoiser_weights(denoiser, ckpt_path, prefixes)
    denoiser.to(device).eval()

    lpips_model, lpips_status = load_lpips_model(args, device)
    if lpips_status:
        print(f"[E1] LPIPS status: {lpips_status}", flush=True)

    methods = parse_method_list(args.methods)
    target_rrs = parse_float_list(args.target_rrs)
    online_deltas = parse_float_list(args.online_deltas)
    method_summaries: List[Dict[str, Any]] = []
    calibration: Dict[str, List[Dict[str, Any]]] = {}

    full_controller = AlwaysRefreshController()
    full_summary = run_method(
        name="full_reference",
        pairs=pairs,
        latent_shape=latent_shape,
        denoiser=denoiser,
        vae=vae,
        conditioner=conditioner,
        sampler=sampler,
        controller=full_controller,
        device=device,
        output_dir=output_dir,
        args=args,
        reference_images=None,
        lpips_model=None,
        return_images=True,
    )
    reference_images = full_summary.pop("final_tensor")
    full_summary.update({"kind": "full", "target_rr": 1.0, "delta": None})
    method_summaries.append(full_summary)

    if "uniform" in methods:
        for target_rr in target_rrs:
            name = f"uniform_rr{format_token(target_rr)}"
            controller = UniformCacheController(target_rr=target_rr)
            summary = run_method(
                name=name,
                pairs=pairs,
                latent_shape=latent_shape,
                denoiser=denoiser,
                vae=vae,
                conditioner=conditioner,
                sampler=sampler,
                controller=controller,
                device=device,
                output_dir=output_dir,
                args=args,
                reference_images=reference_images,
                lpips_model=lpips_model,
                return_images=False,
            )
            summary.update({"kind": "uniform", "target_rr": target_rr, "delta": None})
            method_summaries.append(summary)
            torch.cuda.empty_cache()

    online_metrics = [metric for metric in ["raw", "sea"] if metric in methods]
    calib_pairs = pairs[: max(1, min(args.calib_samples, len(pairs)))]
    if args.calibrate_online:
        for metric in online_metrics:
            rows: List[Dict[str, Any]] = []
            print(f"[E1] Calibrating {metric} on {len(calib_pairs)} samples", flush=True)
            for delta in online_deltas:
                controller = OnlineInputCacheController(
                    metric=metric,
                    delta=delta,
                    warmup_calls=args.warmup_calls,
                    max_skip_calls=args.max_skip_calls,
                    sea_filter_beta=args.sea_filter_beta,
                    sea_filter_eps=args.sea_filter_eps,
                )
                row = run_sampling_only(
                    pairs=calib_pairs,
                    latent_shape=latent_shape,
                    denoiser=denoiser,
                    conditioner=conditioner,
                    sampler=sampler,
                    controller=controller,
                    device=device,
                    args=args,
                )
                row.update({"metric": metric, "delta": delta, "calib_samples": len(calib_pairs)})
                rows.append(row)
                print(f"[E1]   {metric} delta={delta:g} -> RR={row['actual_rr']:.4f}", flush=True)
            calibration[metric] = rows

            for target_rr in target_rrs:
                selected = choose_delta(rows, target_rr)
                delta = float(selected["delta"])
                name = f"{metric}_online_rr{format_token(target_rr)}_delta{format_token(delta)}"
                controller = OnlineInputCacheController(
                    metric=metric,
                    delta=delta,
                    warmup_calls=args.warmup_calls,
                    max_skip_calls=args.max_skip_calls,
                    sea_filter_beta=args.sea_filter_beta,
                    sea_filter_eps=args.sea_filter_eps,
                )
                summary = run_method(
                    name=name,
                    pairs=pairs,
                    latent_shape=latent_shape,
                    denoiser=denoiser,
                    vae=vae,
                    conditioner=conditioner,
                    sampler=sampler,
                    controller=controller,
                    device=device,
                    output_dir=output_dir,
                    args=args,
                    reference_images=reference_images,
                    lpips_model=lpips_model,
                    return_images=False,
                )
                summary.update(
                    {
                        "kind": f"{metric}_online",
                        "target_rr": target_rr,
                        "delta": delta,
                        "calibration_actual_rr": selected["actual_rr"],
                    }
                )
                method_summaries.append(summary)
                torch.cuda.empty_cache()
    else:
        for metric in online_metrics:
            for delta in online_deltas:
                name = f"{metric}_online_delta{format_token(delta)}"
                controller = OnlineInputCacheController(
                    metric=metric,
                    delta=delta,
                    warmup_calls=args.warmup_calls,
                    max_skip_calls=args.max_skip_calls,
                    sea_filter_beta=args.sea_filter_beta,
                    sea_filter_eps=args.sea_filter_eps,
                )
                summary = run_method(
                    name=name,
                    pairs=pairs,
                    latent_shape=latent_shape,
                    denoiser=denoiser,
                    vae=vae,
                    conditioner=conditioner,
                    sampler=sampler,
                    controller=controller,
                    device=device,
                    output_dir=output_dir,
                    args=args,
                    reference_images=reference_images,
                    lpips_model=lpips_model,
                    return_images=False,
                )
                summary.update({"kind": f"{metric}_online", "target_rr": None, "delta": delta})
                method_summaries.append(summary)
                torch.cuda.empty_cache()

    summary_csv = output_dir / "method_summary.csv"
    write_method_summary_csv(method_summaries, summary_csv)

    summary_json = {
        "meta": {
            "config": str(config_path),
            "ckpt": str(ckpt_path),
            "output_dir": str(output_dir),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "num_samples": len(pairs),
            "batch_size": args.batch_size,
            "latent_shape": list(latent_shape),
            "sampler": sampler.__class__.__name__,
            "num_steps": int(sampler.num_steps),
            "denoiser_opportunities_per_sample": total_denoiser_opportunities(sampler),
            "allow_fused_sdpa": bool(args.allow_fused_sdpa),
            "no_autocast": bool(args.no_autocast),
            "lpips_status": lpips_status,
            "load_info": load_info,
            "args": vars(args),
        },
        "calibration": calibration,
        "methods": method_summaries,
        "summary_csv": str(summary_csv),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(json_ready(summary_json), fp, indent=2)

    print(f"[E1] Summary written to {output_dir / 'summary.json'}", flush=True)
    print(f"[E1] Method table written to {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
