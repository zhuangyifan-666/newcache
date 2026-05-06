#!/usr/bin/env python3
"""Single-GPU E0 deterministic full-inference sanity check for PixelGen.

This script intentionally avoids the existing Lightning predict dataloader
because E0 needs fixed class/seed pairs. It runs the same full sampler twice,
saves final images and xhat statistics, then reports deterministic drift.
conda run -n pixelgen python scripts/00_sanity_full_reference.py \
  --device cuda:0 \
  --num-samples 32 \
  --batch-size 1 \
  --preview-count 8 \
  --no-autocast \
  --run-id e0_32samples_fp32

"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.autoencoder.base import fp2uint8  # noqa: E402


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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if force_math_sdpa and torch.cuda.is_available():
        # E0 values reproducibility over speed. PixelGen uses SDPA in JiT attention,
        # and fused CUDA kernels can introduce small run-to-run differences.
        for name, enabled in [
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_cudnn_sdp", False),
            ("enable_math_sdp", True),
        ]:
            fn = getattr(torch.backends.cuda, name, None)
            if fn is not None:
                fn(enabled)


def class_seed_pairs(args: argparse.Namespace) -> List[Tuple[int, int]]:
    classes = parse_int_list(args.classes)
    seeds = parse_int_list(args.seeds)

    if classes is None:
        classes = list(range(args.num_samples))
    if seeds is None:
        seeds = list(range(args.seed_start, args.seed_start + len(classes)))

    if len(classes) != len(seeds):
        raise ValueError("--classes and --seeds must have the same length for E0 pairwise sampling")

    return list(zip(classes, seeds))


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


class TraceStats:
    def __init__(self) -> None:
        self._stats: Dict[str, Dict[int, Dict[str, Any]]] = {
            "predictor": {},
            "corrector": {},
        }

    def update(self, kind: str, step: int, timestep: float, xhat: torch.Tensor) -> None:
        x = xhat.detach().float()
        bucket = self._stats[kind].setdefault(
            step,
            {
                "step": step,
                "timestep": float(timestep),
                "numel": 0,
                "min": float("inf"),
                "max": float("-inf"),
                "sum": 0.0,
                "sumsq": 0.0,
            },
        )
        numel = x.numel()
        bucket["numel"] += int(numel)
        bucket["min"] = min(bucket["min"], float(x.amin().cpu()))
        bucket["max"] = max(bucket["max"], float(x.amax().cpu()))
        bucket["sum"] += float(x.sum().cpu())
        bucket["sumsq"] += float((x * x).sum().cpu())

    def finalize(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for kind, by_step in self._stats.items():
            rows = []
            global_min = float("inf")
            global_max = float("-inf")
            for step in sorted(by_step):
                item = by_step[step]
                n = max(1, item["numel"])
                mean = item["sum"] / n
                var = max(0.0, item["sumsq"] / n - mean * mean)
                row = {
                    "step": item["step"],
                    "timestep": item["timestep"],
                    "min": item["min"],
                    "max": item["max"],
                    "mean": mean,
                    "std": math.sqrt(var),
                }
                rows.append(row)
                global_min = min(global_min, item["min"])
                global_max = max(global_max, item["max"])
            out[kind] = {
                "global_min": None if not rows else global_min,
                "global_max": None if not rows else global_max,
                "per_step": rows,
            }
        return out


@torch.inference_mode()
def heun_jit_with_trace(
    net: nn.Module,
    sampler: nn.Module,
    noise: torch.Tensor,
    condition: torch.Tensor,
    uncondition: torch.Tensor,
    trace_stats: TraceStats,
    collect_predictor_xhat: bool,
) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
    batch_size = noise.shape[0]
    steps = sampler.timesteps.to(noise.device)
    cfg_condition = torch.cat([uncondition, condition], dim=0)
    x = noise
    v_hat, s_hat = 0.0, 0.0
    predictor_xhats: List[torch.Tensor] = []
    denoiser_calls = 0

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
            cfg_x = torch.cat([x, x], dim=0)
            cfg_t_cur = t_cur.repeat(2)
            pred_x = net(cfg_x, cfg_t_cur, cfg_condition)
            denoiser_calls += 1
            out = (pred_x - cfg_x) / (1.0 - cfg_t_cur.view(-1, 1, 1, 1)).clamp_min(sampler.t_eps)
            if t_cur[0] > sampler.guidance_interval_min and t_cur[0] <= sampler.guidance_interval_max:
                out = sampler.guidance_fn(out, sampler.guidance)
            else:
                out = sampler.guidance_fn(out, 1.0)
            v = out
            guided_xhat = x + v * (1.0 - t_cur.view(-1, 1, 1, 1)).clamp_min(sampler.t_eps)
            trace_stats.update("predictor", i, float(t_cur_scalar.detach().cpu()), guided_xhat)
            if collect_predictor_xhat:
                predictor_xhats.append(guided_xhat.detach().cpu())
            s = ((alpha_over_dalpha) * v - x) / (sigma**2 - (alpha_over_dalpha) * dsigma_mul_sigma)
        else:
            v = v_hat
            s = s_hat

        x_hat = sampler.step_fn(x, v, dt, s=s, w=w)

        if i < sampler.num_steps - 1:
            cfg_x_hat = torch.cat([x_hat, x_hat], dim=0)
            cfg_t_hat = t_hat.repeat(2)
            pred_x_hat = net(cfg_x_hat, cfg_t_hat, cfg_condition)
            denoiser_calls += 1
            out = (pred_x_hat - cfg_x_hat) / (1.0 - cfg_t_hat.view(-1, 1, 1, 1)).clamp_min(sampler.t_eps)
            if t_cur[0] > sampler.guidance_interval_min and t_cur[0] <= sampler.guidance_interval_max:
                out = sampler.guidance_fn(out, sampler.guidance)
            else:
                out = sampler.guidance_fn(out, 1.0)

            v_hat = out
            guided_xhat_hat = x_hat + v_hat * (1.0 - t_hat.view(-1, 1, 1, 1)).clamp_min(sampler.t_eps)
            trace_stats.update("corrector", i, float(t_next_scalar.detach().cpu()), guided_xhat_hat)
            s_hat = ((alpha_over_dalpha_hat) * v_hat - x_hat) / (
                sigma_hat**2 - (alpha_over_dalpha_hat) * dsigma_mul_sigma_hat
            )
            v = (v + v_hat) / 2
            s = (s + s_hat) / 2
            x = sampler.step_fn(x, v, dt, s=s, w=w)
        else:
            x = sampler.last_step_fn(x, v, dt, s=s, w=w)

    return x, predictor_xhats, denoiser_calls


def run_once(
    name: str,
    pairs: Sequence[Tuple[int, int]],
    latent_shape: Sequence[int],
    denoiser: nn.Module,
    vae: nn.Module,
    conditioner: nn.Module,
    sampler: nn.Module,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
    save_previews: bool,
) -> Dict[str, Any]:
    trace_stats = TraceStats()
    final_images: List[torch.Tensor] = []
    denoiser_calls = 0
    start_time = time.perf_counter()
    preview_by_step: Dict[int, List[torch.Tensor]] = {}
    preview_seen = 0

    run_dir = output_dir / name
    final_dir = run_dir / "final_images"
    final_dir.mkdir(parents=True, exist_ok=True)

    autocast_dtype = torch.bfloat16 if args.autocast_dtype == "bf16" else torch.float16

    for batch_start in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[batch_start : batch_start + args.batch_size]
        labels = [pair[0] for pair in batch_pairs]
        noise = make_noise_batch(batch_pairs, latent_shape, device, args.noise_scale)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=not args.no_autocast):
            condition, uncondition = conditioner(labels, {})
            collect = save_previews and preview_seen < args.preview_count
            samples, predictor_xhats, calls = heun_jit_with_trace(
                denoiser,
                sampler,
                noise,
                condition,
                uncondition,
                trace_stats,
                collect_predictor_xhat=collect,
            )
            decoded = vae.decode(samples)

        denoiser_calls += calls
        decoded_cpu = decoded.detach().cpu().float()
        final_images.append(decoded_cpu)

        for local_idx, (class_id, seed) in enumerate(batch_pairs):
            save_tensor_image(
                decoded_cpu[local_idx],
                final_dir / f"class{class_id:04d}_seed{seed}.png",
            )

        if save_previews and predictor_xhats and preview_seen < args.preview_count:
            remaining = args.preview_count - preview_seen
            take = min(remaining, decoded_cpu.shape[0])
            for step, xhat in enumerate(predictor_xhats):
                preview_by_step.setdefault(step, []).append(xhat[:take])
            preview_seen += take

    if save_previews and preview_by_step:
        preview_dir = run_dir / "xhat_previews"
        for step, chunks in preview_by_step.items():
            save_image_grid(
                torch.cat(chunks, dim=0),
                preview_dir / f"predictor_step{step:03d}.png",
                resize=args.preview_size,
                columns=args.preview_columns,
            )

    elapsed = time.perf_counter() - start_time
    final_tensor = torch.cat(final_images, dim=0) if final_images else torch.empty(0)
    trace_summary = trace_stats.finalize()
    with open(run_dir / "xhat_trace_stats.json", "w", encoding="utf-8") as fp:
        json.dump(trace_summary, fp, indent=2)

    return {
        "name": name,
        "elapsed_sec": elapsed,
        "denoiser_calls": denoiser_calls,
        "final_tensor": final_tensor,
        "trace_summary": trace_summary,
    }


def compare_runs(
    pairs: Sequence[Tuple[int, int]],
    run_a: torch.Tensor,
    run_b: torch.Tensor,
    output_dir: Path,
    max_abs_tol: float,
    min_psnr: float,
) -> Dict[str, Any]:
    rows = []
    max_abs_values = []
    psnr_values = []
    uint8_equal_count = 0

    for idx, (class_id, seed) in enumerate(pairs):
        a = run_a[idx]
        b = run_b[idx]
        diff = (a - b).float()
        mse = float((diff * diff).mean())
        max_abs = float(diff.abs().max())
        mean_abs = float(diff.abs().mean())
        psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(2.0 / math.sqrt(mse))
        a_u8 = fp2uint8(a.unsqueeze(0).clone())[0]
        b_u8 = fp2uint8(b.unsqueeze(0).clone())[0]
        uint8_equal = bool(torch.equal(a_u8, b_u8))
        uint8_equal_count += int(uint8_equal)
        max_abs_values.append(max_abs)
        psnr_values.append(psnr)
        rows.append(
            {
                "index": idx,
                "class_id": class_id,
                "seed": seed,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "mse": mse,
                "psnr_db": psnr,
                "uint8_equal": uint8_equal,
            }
        )

    csv_path = output_dir / "per_sample_determinism.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    finite_psnr = [value for value in psnr_values if math.isfinite(value)]
    min_observed_psnr = min(finite_psnr) if finite_psnr else float("inf")
    max_observed_abs = max(max_abs_values) if max_abs_values else 0.0
    passed = max_observed_abs <= max_abs_tol or min_observed_psnr >= min_psnr

    return {
        "passed": passed,
        "num_samples": len(pairs),
        "max_abs": max_observed_abs,
        "mean_max_abs": sum(max_abs_values) / max(1, len(max_abs_values)),
        "min_psnr_db": min_observed_psnr,
        "mean_psnr_db": (
            float("inf")
            if not finite_psnr
            else sum(finite_psnr) / max(1, len(finite_psnr))
        ),
        "uint8_equal_count": uint8_equal_count,
        "uint8_equal_fraction": uint8_equal_count / max(1, len(pairs)),
        "per_sample_csv": str(csv_path),
    }


def write_pairs_csv(pairs: Sequence[Tuple[int, int]], output_dir: Path) -> None:
    with open(output_dir / "class_seed_pairs.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["index", "class_id", "seed"])
        writer.writeheader()
        for idx, (class_id, seed) in enumerate(pairs):
            writer.writerow({"index": idx, "class_id": class_id, "seed": seed})


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs_c2i/PixelGen_XL_without_CFG.yaml")
    parser.add_argument("--ckpt", default="ckpts/PixelGen_XL_80ep.ckpt")
    parser.add_argument("--output-dir", default="outputs/e0_sanity")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--classes", default=None, help="Comma-separated class ids. Defaults to 0..num_samples-1.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to seed_start..")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--autocast-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--allow-fused-sdpa", action="store_true")
    parser.add_argument("--global-seed", type=int, default=2026)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--preview-size", type=int, default=128)
    parser.add_argument("--preview-columns", type=int, default=4)
    parser.add_argument("--max-abs-tol", type=float, default=1e-5)
    parser.add_argument("--min-psnr", type=float, default=80.0)
    parser.add_argument(
        "--weight-prefixes",
        default="ema_denoiser.,denoiser.,",
        help="Comma-separated checkpoint prefixes to try in order.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. E0 requires a GPU-visible shell.")

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

    pairs = class_seed_pairs(args)
    if not pairs:
        raise ValueError("No class/seed pairs selected for this shard")
    write_pairs_csv(pairs, output_dir)

    config = OmegaConf.load(config_path)
    latent_shape = config.data.pred_dataset.init_args.latent_shape
    vae = instantiate_from_config(config.model.vae).to(device).eval()
    denoiser = instantiate_from_config(config.model.denoiser).to(device).eval()
    conditioner = instantiate_from_config(config.model.conditioner).to(device).eval()
    sampler = instantiate_from_config(config.model.diffusion_sampler).to(device).eval()

    prefixes = [item for item in args.weight_prefixes.split(",") if item != ""]
    if args.weight_prefixes.endswith(","):
        prefixes.append("")
    load_info = load_denoiser_weights(denoiser, ckpt_path, prefixes)
    denoiser.to(device).eval()

    common_meta = {
        "config": str(config_path),
        "ckpt": str(ckpt_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "allow_fused_sdpa": bool(args.allow_fused_sdpa),
        "num_pairs": len(pairs),
        "latent_shape": list(latent_shape),
        "sampler": sampler.__class__.__name__,
        "num_steps": int(sampler.num_steps),
        "guidance": float(sampler.guidance),
        "timeshift": float(getattr(sampler, "timeshift", -1.0)),
        "exact_henu": bool(getattr(sampler, "exact_henu", False)),
        "weight_load": load_info,
        "args": vars(args),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(common_meta, fp, indent=2)

    print(f"[E0] Output: {output_dir}")
    print(f"[E0] Running {len(pairs)} fixed class/seed pairs twice on {device}")
    print(f"[E0] Loaded weights: {load_info['loaded']}/{load_info['total']} from {load_info['used_prefixes']}")

    run_a = run_once(
        "run_a",
        pairs,
        latent_shape,
        denoiser,
        vae,
        conditioner,
        sampler,
        device,
        output_dir,
        args,
        save_previews=args.preview_count > 0,
    )
    torch.cuda.empty_cache()
    run_b = run_once(
        "run_b",
        pairs,
        latent_shape,
        denoiser,
        vae,
        conditioner,
        sampler,
        device,
        output_dir,
        args,
        save_previews=False,
    )

    comparison = compare_runs(
        pairs,
        run_a["final_tensor"],
        run_b["final_tensor"],
        output_dir,
        max_abs_tol=args.max_abs_tol,
        min_psnr=args.min_psnr,
    )
    summary = {
        **common_meta,
        "run_a": {
            "elapsed_sec": run_a["elapsed_sec"],
            "denoiser_calls": run_a["denoiser_calls"],
            "xhat_predictor_global_min": run_a["trace_summary"]["predictor"]["global_min"],
            "xhat_predictor_global_max": run_a["trace_summary"]["predictor"]["global_max"],
            "xhat_corrector_global_min": run_a["trace_summary"]["corrector"]["global_min"],
            "xhat_corrector_global_max": run_a["trace_summary"]["corrector"]["global_max"],
        },
        "run_b": {
            "elapsed_sec": run_b["elapsed_sec"],
            "denoiser_calls": run_b["denoiser_calls"],
            "xhat_predictor_global_min": run_b["trace_summary"]["predictor"]["global_min"],
            "xhat_predictor_global_max": run_b["trace_summary"]["predictor"]["global_max"],
            "xhat_corrector_global_min": run_b["trace_summary"]["corrector"]["global_min"],
            "xhat_corrector_global_max": run_b["trace_summary"]["corrector"]["global_max"],
        },
        "comparison": comparison,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("[E0] Summary")
    print(json.dumps(summary["comparison"], indent=2))
    print(f"[E0] {'PASS' if comparison['passed'] else 'WARN'}")


if __name__ == "__main__":
    main()
