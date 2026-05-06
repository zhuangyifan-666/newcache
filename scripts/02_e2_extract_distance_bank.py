#!/usr/bin/env python3
"""Extract E2 oracle distance bank from full PixelGen trajectories.

E2 is not a cache rerun. It runs the full uncached sampler and records
adjacent-call distance sequences for later oracle schedule analysis.

Recommended E2 main:

CUDA_VISIBLE_DEVICES=0 conda run -n pixelgen python scripts/02_e2_extract_distance_bank.py \
  --device cuda:0 \
  --num-samples 256 \
  --no-autocast \
  --run-id e2_main_256_fp32
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion.flow_matching.e1_cache import (  # noqa: E402
    apply_sea_filter,
    extract_jit_modulated_proxy,
    relative_l1_distance,
    unwrap_runtime_module,
)
from src.models.autoencoder.base import fp2uint8  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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
    np.random.seed(seed)
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


def make_noise(
    seed: int,
    latent_shape: Sequence[int],
    device: torch.device,
    noise_scale: float,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = noise_scale * torch.randn(tuple(latent_shape), generator=generator, dtype=torch.float32)
    return noise.unsqueeze(0).to(device)


def total_denoiser_opportunities(sampler: nn.Module) -> int:
    if getattr(sampler, "exact_henu", False):
        return int(2 * sampler.num_steps - 1)
    return int(sampler.num_steps)


def image_to_uint8_hwc(image: torch.Tensor) -> Any:
    image_u8 = fp2uint8(image.detach().cpu().unsqueeze(0).clone())[0]
    return image_u8.permute(1, 2, 0).numpy()


def save_tensor_image(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_to_uint8_hwc(image)).save(path)


def load_dino_model(args: argparse.Namespace, device: torch.device) -> Tuple[Optional[nn.Module], str]:
    if args.skip_dino:
        return None, "skipped"

    repo = Path(args.dino_repo).expanduser()
    try:
        if repo.exists():
            model = torch.hub.load(str(repo), args.dino_model, source="local")
            source = str(repo)
        else:
            model = torch.hub.load(args.dino_repo, args.dino_model)
            source = args.dino_repo
        model.to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model, source
    except Exception as exc:
        raise RuntimeError(f"Failed to load DINO model from {args.dino_repo}: {exc}") from exc


def load_lpips_model(args: argparse.Namespace, device: torch.device) -> Tuple[Optional[nn.Module], str]:
    if args.skip_lpips:
        return None, "skipped"
    try:
        import lpips  # type: ignore

        model = lpips.LPIPS(net=args.lpips_net).to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model, args.lpips_net
    except Exception as exc:
        raise RuntimeError(f"Failed to load LPIPS-{args.lpips_net}: {exc}") from exc


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def compute_dino_distances(
    xhats: torch.Tensor,
    dino_model: Optional[nn.Module],
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    num_calls = xhats.shape[0]
    if dino_model is None:
        return np.full((num_calls - 1,), np.nan, dtype=np.float32)

    features: List[torch.Tensor] = []
    for start in range(0, num_calls, args.dino_batch_size):
        batch = xhats[start : start + args.dino_batch_size].to(device=device, dtype=torch.float32)
        batch = ((batch.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
        if args.dino_size > 0:
            batch = F.interpolate(batch, size=(args.dino_size, args.dino_size), mode="bicubic", align_corners=False)
        batch = normalize_imagenet(batch)
        out = dino_model.forward_features(batch)
        if args.dino_feature == "cls":
            feat = out["x_norm_clstoken"]
        else:
            feat = out["x_norm_patchtokens"].mean(dim=1)
        features.append(F.normalize(feat.float(), dim=-1).detach().cpu())

    feats = torch.cat(features, dim=0)
    distances = 1.0 - (feats[:-1] * feats[1:]).sum(dim=-1)
    return distances.float().numpy().astype(np.float32)


@torch.inference_mode()
def compute_lpips_distances(
    xhats: torch.Tensor,
    lpips_model: Optional[nn.Module],
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    num_calls = xhats.shape[0]
    if lpips_model is None:
        return np.full((num_calls - 1,), np.nan, dtype=np.float32)

    values: List[torch.Tensor] = []
    prev = xhats[:-1]
    cur = xhats[1:]
    for start in range(0, num_calls - 1, args.lpips_batch_size):
        a = prev[start : start + args.lpips_batch_size].to(device=device, dtype=torch.float32).clamp(-1, 1)
        b = cur[start : start + args.lpips_batch_size].to(device=device, dtype=torch.float32).clamp(-1, 1)
        if args.lpips_size > 0:
            a = F.interpolate(a, size=(args.lpips_size, args.lpips_size), mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=(args.lpips_size, args.lpips_size), mode="bilinear", align_corners=False)
        values.append(lpips_model(a, b).view(-1).detach().float().cpu())
    return torch.cat(values, dim=0).numpy().astype(np.float32)


@torch.inference_mode()
def run_full_sample(
    *,
    sample_index: int,
    class_id: int,
    seed: int,
    latent_shape: Sequence[int],
    denoiser: nn.Module,
    conditioner: nn.Module,
    sampler: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    del sample_index
    noise = make_noise(seed, latent_shape, device, args.noise_scale)
    condition, uncondition = conditioner([class_id], {})
    cfg_condition = torch.cat([uncondition, condition], dim=0)
    total_calls = total_denoiser_opportunities(sampler)

    raw_distances: List[float] = []
    sea_distances: List[float] = []
    xhats: List[torch.Tensor] = []
    call_timesteps: List[float] = []
    call_steps: List[int] = []
    call_kinds: List[int] = []

    previous_raw_proxy: Optional[torch.Tensor] = None
    previous_sea_proxy: Optional[torch.Tensor] = None
    runtime_model = unwrap_runtime_module(denoiser)

    denoiser_time = 0.0

    def denoiser_call(
        state_x: torch.Tensor,
        t: torch.Tensor,
        guidance_t: torch.Tensor,
        step_index: int,
        kind: int,
    ) -> torch.Tensor:
        nonlocal previous_raw_proxy, previous_sea_proxy, denoiser_time

        cfg_x = torch.cat([state_x, state_x], dim=0)
        cfg_t = t.repeat(2)
        raw_proxy = extract_jit_modulated_proxy(runtime_model, cfg_x, cfg_t, cfg_condition)
        sea_proxy = apply_sea_filter(raw_proxy, cfg_t, beta=args.sea_filter_beta, eps=args.sea_filter_eps)

        if previous_raw_proxy is not None and previous_sea_proxy is not None:
            raw_distances.append(relative_l1_distance(raw_proxy, previous_raw_proxy, eps=args.sea_filter_eps))
            sea_distances.append(relative_l1_distance(sea_proxy, previous_sea_proxy, eps=args.sea_filter_eps))

        previous_raw_proxy = raw_proxy.detach()
        previous_sea_proxy = sea_proxy.detach()

        if cfg_x.is_cuda:
            torch.cuda.synchronize(cfg_x.device)
        start = time.perf_counter()
        cfg_out = denoiser(cfg_x, cfg_t, cfg_condition)
        if cfg_x.is_cuda:
            torch.cuda.synchronize(cfg_x.device)
        denoiser_time += time.perf_counter() - start

        v_all = (cfg_out - cfg_x) / (1.0 - cfg_t.view(-1, 1, 1, 1)).clamp_min(sampler.t_eps)
        if guidance_t[0] > sampler.guidance_interval_min and guidance_t[0] <= sampler.guidance_interval_max:
            v = sampler.guidance_fn(v_all, sampler.guidance)
        else:
            v = sampler.guidance_fn(v_all, 1.0)

        guided_xhat = state_x + v * (1.0 - t.view(-1, 1, 1, 1)).clamp_min(sampler.t_eps)
        xhats.append(guided_xhat.detach().cpu().float())
        call_timesteps.append(float(t[0].detach().cpu()))
        call_steps.append(int(step_index))
        call_kinds.append(int(kind))
        return v

    steps = sampler.timesteps.to(noise.device)
    x = noise
    v_hat, s_hat = 0.0, 0.0
    call_count = 0
    sample_start = time.perf_counter()

    autocast_dtype = torch.bfloat16 if args.autocast_dtype == "bf16" else torch.float16
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=not args.no_autocast):
        for i, (t_cur_scalar, t_next_scalar) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next_scalar - t_cur_scalar
            t_cur = t_cur_scalar.repeat(1).to(noise.device, noise.dtype)
            sigma = sampler.scheduler.sigma(t_cur)
            alpha_over_dalpha = 1 / sampler.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = sampler.scheduler.dsigma_mul_sigma(t_cur)
            t_hat = t_next_scalar.repeat(1).to(noise.device, noise.dtype)
            sigma_hat = sampler.scheduler.sigma(t_hat)
            alpha_over_dalpha_hat = 1 / sampler.scheduler.dalpha_over_alpha(t_hat)
            dsigma_mul_sigma_hat = sampler.scheduler.dsigma_mul_sigma(t_hat)
            w = sampler.w_scheduler.w(t_cur) if sampler.w_scheduler else 0.0

            if i == 0 or sampler.exact_henu:
                v = denoiser_call(x, t_cur, t_cur, i, 0)
                call_count += 1
                s = ((alpha_over_dalpha) * v - x) / (sigma**2 - (alpha_over_dalpha) * dsigma_mul_sigma)
            else:
                v = v_hat
                s = s_hat

            x_hat_state = sampler.step_fn(x, v, dt, s=s, w=w)

            if i < sampler.num_steps - 1:
                v_hat = denoiser_call(x_hat_state, t_hat, t_cur, i, 1)
                call_count += 1
                s_hat = ((alpha_over_dalpha_hat) * v_hat - x_hat_state) / (
                    sigma_hat**2 - (alpha_over_dalpha_hat) * dsigma_mul_sigma_hat
                )
                v = (v + v_hat) / 2
                s = (s + s_hat) / 2
                x = sampler.step_fn(x, v, dt, s=s, w=w)
            else:
                x = sampler.last_step_fn(x, v, dt, s=s, w=w)

    if call_count != total_calls:
        raise RuntimeError(f"Expected {total_calls} denoiser calls, got {call_count}.")
    if len(raw_distances) != total_calls - 1 or len(sea_distances) != total_calls - 1:
        raise RuntimeError("Distance sequence length mismatch.")

    xhat_tensor = torch.cat(xhats, dim=0)
    return {
        "raw": np.asarray(raw_distances, dtype=np.float32),
        "sea": np.asarray(sea_distances, dtype=np.float32),
        "xhats": xhat_tensor,
        "final": x.detach().cpu().float(),
        "call_timesteps": np.asarray(call_timesteps, dtype=np.float32),
        "call_steps": np.asarray(call_steps, dtype=np.int16),
        "call_kinds": np.asarray(call_kinds, dtype=np.int8),
        "sample_time_sec": time.perf_counter() - sample_start,
        "denoiser_time_sec": denoiser_time,
    }


def init_bank(num_samples: int, num_distances: int) -> Dict[str, np.ndarray]:
    return {
        "raw": np.full((num_samples, num_distances), np.nan, dtype=np.float32),
        "sea": np.full((num_samples, num_distances), np.nan, dtype=np.float32),
        "dino": np.full((num_samples, num_distances), np.nan, dtype=np.float32),
        "lpips": np.full((num_samples, num_distances), np.nan, dtype=np.float32),
        "sample_time_sec": np.full((num_samples,), np.nan, dtype=np.float32),
        "denoiser_time_sec": np.full((num_samples,), np.nan, dtype=np.float32),
        "completed": np.zeros((num_samples,), dtype=np.bool_),
    }


def load_or_init_bank(output_dir: Path, num_samples: int, num_distances: int, resume: bool) -> Dict[str, np.ndarray]:
    partial_path = output_dir / "distance_bank_partial.npz"
    final_path = output_dir / "distance_bank.npz"
    load_path = partial_path if partial_path.exists() else final_path
    if resume and load_path.exists():
        loaded = np.load(load_path)
        bank = {key: loaded[key] for key in loaded.files if key in init_bank(num_samples, num_distances)}
        fresh = init_bank(num_samples, num_distances)
        for key, value in bank.items():
            if fresh[key].shape == value.shape:
                fresh[key] = value
        return fresh
    return init_bank(num_samples, num_distances)


def save_bank(
    path: Path,
    bank: Dict[str, np.ndarray],
    call_timesteps: np.ndarray,
    call_steps: np.ndarray,
    call_kinds: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        raw=bank["raw"],
        sea=bank["sea"],
        dino=bank["dino"],
        lpips=bank["lpips"],
        sample_time_sec=bank["sample_time_sec"],
        denoiser_time_sec=bank["denoiser_time_sec"],
        completed=bank["completed"],
        call_timesteps=call_timesteps,
        call_steps=call_steps,
        call_kinds=call_kinds,
    )


def write_average_curves(output_dir: Path, bank: Dict[str, np.ndarray], call_timesteps, call_steps, call_kinds) -> None:
    path = output_dir / "per_call_average_curves.csv"
    completed = bank["completed"]
    fields = [
        "distance_index",
        "prev_call_step",
        "prev_call_kind",
        "prev_timestep",
        "cur_call_step",
        "cur_call_kind",
        "cur_timestep",
        "raw_mean",
        "raw_std",
        "sea_mean",
        "sea_std",
        "dino_mean",
        "dino_std",
        "lpips_mean",
        "lpips_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for idx in range(bank["raw"].shape[1]):
            row = {
                "distance_index": idx,
                "prev_call_step": int(call_steps[idx]),
                "prev_call_kind": int(call_kinds[idx]),
                "prev_timestep": float(call_timesteps[idx]),
                "cur_call_step": int(call_steps[idx + 1]),
                "cur_call_kind": int(call_kinds[idx + 1]),
                "cur_timestep": float(call_timesteps[idx + 1]),
            }
            for key in ["raw", "sea", "dino", "lpips"]:
                values = bank[key][completed, idx]
                row[f"{key}_mean"] = float(np.nanmean(values)) if values.size else math.nan
                row[f"{key}_std"] = float(np.nanstd(values)) if values.size else math.nan
            writer.writerow(row)


def write_progress_json(
    output_dir: Path,
    meta: Dict[str, Any],
    bank: Dict[str, np.ndarray],
    started_at: float,
) -> None:
    completed_count = int(bank["completed"].sum())
    payload = {
        "meta": meta,
        "completed_samples": completed_count,
        "total_samples": int(bank["completed"].shape[0]),
        "elapsed_sec": time.perf_counter() - started_at,
        "nan_counts": {
            key: int(np.isnan(bank[key][bank["completed"]]).sum())
            for key in ["raw", "sea", "dino", "lpips"]
        },
        "mean_sample_time_sec": float(np.nanmean(bank["sample_time_sec"])) if completed_count else None,
        "mean_denoiser_time_sec": float(np.nanmean(bank["denoiser_time_sec"])) if completed_count else None,
    }
    with open(output_dir / "progress.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs_c2i/PixelGen_XL_without_CFG.yaml")
    parser.add_argument("--ckpt", default="ckpts/PixelGen_XL_80ep.ckpt")
    parser.add_argument("--output-dir", default="outputs/e2_distance_bank")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--classes", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--autocast-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--allow-fused-sdpa", action="store_true")
    parser.add_argument("--global-seed", type=int, default=2026)
    parser.add_argument("--sea-filter-beta", type=float, default=2.0)
    parser.add_argument("--sea-filter-eps", type=float, default=1e-6)
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--dino-repo", default="/root/.cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--dino-model", default="dinov2_vitb14")
    parser.add_argument("--dino-feature", choices=["patch_mean", "cls"], default="patch_mean")
    parser.add_argument("--dino-size", type=int, default=224)
    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--lpips-size", type=int, default=128)
    parser.add_argument("--lpips-batch-size", type=int, default=16)
    parser.add_argument("--save-preview-count", type=int, default=16)
    parser.add_argument("--save-every-samples", type=int, default=8)
    parser.add_argument(
        "--weight-prefixes",
        default="ema_denoiser.,denoiser.,",
        help="Comma-separated checkpoint prefixes to try in order.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. E2 requires a GPU-visible shell.")
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

    dino_model, dino_source = load_dino_model(args, device)
    lpips_model, lpips_source = load_lpips_model(args, device)

    total_calls = total_denoiser_opportunities(sampler)
    num_distances = total_calls - 1
    bank = load_or_init_bank(output_dir, len(pairs), num_distances, args.resume)
    call_timesteps = np.full((total_calls,), np.nan, dtype=np.float32)
    call_steps = np.full((total_calls,), -1, dtype=np.int16)
    call_kinds = np.full((total_calls,), -1, dtype=np.int8)

    meta = {
        "config": str(config_path),
        "ckpt": str(ckpt_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "num_samples": len(pairs),
        "latent_shape": list(latent_shape),
        "sampler": sampler.__class__.__name__,
        "num_steps": int(sampler.num_steps),
        "denoiser_opportunities_per_sample": total_calls,
        "distances_per_sample": num_distances,
        "no_autocast": bool(args.no_autocast),
        "allow_fused_sdpa": bool(args.allow_fused_sdpa),
        "dino_source": dino_source,
        "dino_feature": args.dino_feature,
        "dino_size": args.dino_size,
        "lpips_source": lpips_source,
        "lpips_size": args.lpips_size,
        "load_info": load_info,
        "args": vars(args),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    started_at = time.perf_counter()
    print(
        f"[E2] Extracting distance bank: samples={len(pairs)}, calls/sample={total_calls}, "
        f"distances/sample={num_distances}",
        flush=True,
    )

    preview_dir = output_dir / "preview_final_images"
    for idx, (class_id, seed) in enumerate(pairs):
        if bool(bank["completed"][idx]):
            print(f"[E2] Skip completed sample {idx + 1}/{len(pairs)}", flush=True)
            continue

        sample = run_full_sample(
            sample_index=idx,
            class_id=class_id,
            seed=seed,
            latent_shape=latent_shape,
            denoiser=denoiser,
            conditioner=conditioner,
            sampler=sampler,
            device=device,
            args=args,
        )
        dino_dist = compute_dino_distances(sample["xhats"], dino_model, device, args)
        lpips_dist = compute_lpips_distances(sample["xhats"], lpips_model, device, args)

        bank["raw"][idx] = sample["raw"]
        bank["sea"][idx] = sample["sea"]
        bank["dino"][idx] = dino_dist
        bank["lpips"][idx] = lpips_dist
        bank["sample_time_sec"][idx] = float(sample["sample_time_sec"])
        bank["denoiser_time_sec"][idx] = float(sample["denoiser_time_sec"])
        bank["completed"][idx] = True

        if np.isnan(call_timesteps).all():
            call_timesteps = sample["call_timesteps"]
            call_steps = sample["call_steps"]
            call_kinds = sample["call_kinds"]

        if idx < args.save_preview_count:
            decoded = vae.decode(sample["final"].to(device)).detach().cpu()[0]
            save_tensor_image(decoded, preview_dir / f"class{class_id:04d}_seed{seed}.png")

        del sample
        torch.cuda.empty_cache()

        completed = int(bank["completed"].sum())
        print(
            f"[E2] Sample {idx + 1}/{len(pairs)} done "
            f"({completed}/{len(pairs)} completed). "
            f"last_sample_time={bank['sample_time_sec'][idx]:.2f}s",
            flush=True,
        )

        if completed % max(1, args.save_every_samples) == 0 or completed == len(pairs):
            save_bank(output_dir / "distance_bank_partial.npz", bank, call_timesteps, call_steps, call_kinds)
            write_progress_json(output_dir, meta, bank, started_at)
            write_average_curves(output_dir, bank, call_timesteps, call_steps, call_kinds)
            print(f"[E2] Partial bank saved after {completed} samples", flush=True)

    final_path = output_dir / "distance_bank.npz"
    save_bank(final_path, bank, call_timesteps, call_steps, call_kinds)
    write_progress_json(output_dir, meta, bank, started_at)
    write_average_curves(output_dir, bank, call_timesteps, call_steps, call_kinds)

    summary = {
        "meta": meta,
        "completed_samples": int(bank["completed"].sum()),
        "total_samples": len(pairs),
        "elapsed_sec": time.perf_counter() - started_at,
        "distance_bank": str(final_path),
        "average_curves": str(output_dir / "per_call_average_curves.csv"),
        "nan_counts": {
            key: int(np.isnan(bank[key][bank["completed"]]).sum())
            for key in ["raw", "sea", "dino", "lpips"]
        },
        "metric_means": {
            key: float(np.nanmean(bank[key][bank["completed"]]))
            for key in ["raw", "sea", "dino", "lpips"]
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[E2] Final distance bank written to {final_path}", flush=True)
    print(f"[E2] Summary written to {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
