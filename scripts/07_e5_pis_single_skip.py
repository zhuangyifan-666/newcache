#!/usr/bin/env python3
"""E5 single-skip Perceptual Intervention Score bank for PixelGen.

E5 asks a causal question: if exactly one denoiser call is skipped while all
later calls are refreshed normally, how much does the final image change?

This script records a full reference trajectory for each sample, then replays
only the suffix after each selected intervention call. That keeps the
intervention faithful to the cache semantics while avoiding a full resample
from call 0 for every single call.

Recommended minimal pilot from the reboot plan:

conda run -n pixelgen python scripts/07_e5_pis_single_skip.py \
  --device cuda:0 \
  --num-samples 4 \
  --call-stride 2 \
  --no-autocast \
  --run-id e5_min4_stride2_fp32
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_helper_module(filename: str, module_name: str) -> Any:
    path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E1 = _load_helper_module("01_e1_online_cache.py", "pixelgen_e1_helpers")
E2 = _load_helper_module("02_e2_extract_distance_bank.py", "pixelgen_e2_helpers")


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        return json_ready(value.detach().cpu().tolist())
    return value


def safe_mean(values: np.ndarray) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def safe_min(values: np.ndarray) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.min())


def safe_max(values: np.ndarray) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.max())


def call_kind_name(kind: int) -> str:
    return "predictor" if int(kind) == 0 else "corrector"


def build_call_metadata(sampler: nn.Module) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = sampler.timesteps.detach().cpu().float()
    call_timesteps: List[float] = []
    call_steps: List[int] = []
    call_kinds: List[int] = []
    for step_idx, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
        call_timesteps.append(float(t_cur))
        call_steps.append(int(step_idx))
        call_kinds.append(0)
        if step_idx < int(sampler.num_steps) - 1:
            call_timesteps.append(float(t_next))
            call_steps.append(int(step_idx))
            call_kinds.append(1)
    return (
        np.asarray(call_timesteps, dtype=np.float32),
        np.asarray(call_steps, dtype=np.int16),
        np.asarray(call_kinds, dtype=np.int8),
    )


def selected_call_mask(total_calls: int, args: argparse.Namespace) -> np.ndarray:
    mask = np.zeros((total_calls,), dtype=bool)
    explicit = parse_int_list(args.call_indices)
    if explicit is not None:
        for idx in explicit:
            if idx < 0 or idx >= total_calls:
                raise ValueError(f"call index {idx} is outside [0, {total_calls})")
            mask[idx] = True
        return mask

    stride = max(1, int(args.call_stride))
    mask[::stride] = True
    return mask


def _cfg_denoiser_call(
    net: nn.Module,
    state_x: torch.Tensor,
    t: torch.Tensor,
    cfg_condition: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg_x = torch.cat([state_x, state_x], dim=0)
    cfg_t = t.repeat(2)
    cfg_out = net(cfg_x, cfg_t, cfg_condition)
    return cfg_out, cfg_x, cfg_t


def _guided_velocity(
    sampler: nn.Module,
    cfg_out: torch.Tensor,
    cfg_x: torch.Tensor,
    cfg_t: torch.Tensor,
    guidance_t: torch.Tensor,
) -> torch.Tensor:
    v_all = (cfg_out - cfg_x) / (1.0 - cfg_t.view(-1, 1, 1, 1)).clamp_min(sampler.t_eps)
    if guidance_t[0] > sampler.guidance_interval_min and guidance_t[0] <= sampler.guidance_interval_max:
        return sampler.guidance_fn(v_all, sampler.guidance)
    return sampler.guidance_fn(v_all, 1.0)


def _score_s(sampler: nn.Module, t: torch.Tensor, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    sigma = sampler.scheduler.sigma(t)
    alpha_over_dalpha = 1 / sampler.scheduler.dalpha_over_alpha(t)
    dsigma_mul_sigma = sampler.scheduler.dsigma_mul_sigma(t)
    return ((alpha_over_dalpha) * v - x) / (sigma**2 - (alpha_over_dalpha) * dsigma_mul_sigma)


@torch.inference_mode()
def run_full_trace(
    *,
    net: nn.Module,
    sampler: nn.Module,
    noise: torch.Tensor,
    condition: torch.Tensor,
    uncondition: torch.Tensor,
) -> Dict[str, Any]:
    batch_size = noise.shape[0]
    if batch_size != 1:
        raise ValueError("E5 single-skip trace currently expects batch_size=1.")

    steps = sampler.timesteps.to(noise.device)
    cfg_condition = torch.cat([uncondition, condition], dim=0)
    x = noise

    step_start_xs: List[torch.Tensor] = []
    call_outputs: List[torch.Tensor] = []
    denoiser_calls = 0

    for step_idx, (t_cur_scalar, t_next_scalar) in enumerate(zip(steps[:-1], steps[1:])):
        step_start_xs.append(x.detach().cpu().float())

        dt = t_next_scalar - t_cur_scalar
        t_cur = t_cur_scalar.repeat(batch_size).to(noise.device, noise.dtype)
        t_hat = t_next_scalar.repeat(batch_size).to(noise.device, noise.dtype)
        w = sampler.w_scheduler.w(t_cur) if sampler.w_scheduler else 0.0

        cfg_out, cfg_x, cfg_t = _cfg_denoiser_call(net, x, t_cur, cfg_condition)
        denoiser_calls += 1
        call_outputs.append(cfg_out.detach().cpu().float())
        v = _guided_velocity(sampler, cfg_out, cfg_x, cfg_t, guidance_t=t_cur)
        s = _score_s(sampler, t_cur, x, v)
        x_hat_state = sampler.step_fn(x, v, dt, s=s, w=w)

        if step_idx < int(sampler.num_steps) - 1:
            cfg_out_hat, cfg_x_hat, cfg_t_hat = _cfg_denoiser_call(net, x_hat_state, t_hat, cfg_condition)
            denoiser_calls += 1
            call_outputs.append(cfg_out_hat.detach().cpu().float())
            v_hat = _guided_velocity(sampler, cfg_out_hat, cfg_x_hat, cfg_t_hat, guidance_t=t_cur)
            s_hat = _score_s(sampler, t_hat, x_hat_state, v_hat)
            v_avg = (v + v_hat) / 2
            s_avg = (s + s_hat) / 2
            x = sampler.step_fn(x, v_avg, dt, s=s_avg, w=w)
        else:
            x = sampler.last_step_fn(x, v, dt, s=s, w=w)

    return {
        "final": x.detach().cpu().float(),
        "step_start_xs": step_start_xs,
        "call_outputs": call_outputs,
        "denoiser_calls": denoiser_calls,
    }


@torch.inference_mode()
def run_single_skip_suffix(
    *,
    net: nn.Module,
    sampler: nn.Module,
    trace: Mapping[str, Any],
    condition: torch.Tensor,
    uncondition: torch.Tensor,
    skip_call_index: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    if skip_call_index <= 0:
        raise ValueError("call 0 cannot be skipped because no cached output exists yet.")

    total_calls = len(trace["call_outputs"])
    if skip_call_index >= total_calls:
        raise ValueError(f"skip_call_index {skip_call_index} outside total calls {total_calls}")

    step_index = skip_call_index // 2
    batch_size = 1
    steps = sampler.timesteps.to(device)
    cfg_condition = torch.cat([uncondition, condition], dim=0)
    x = trace["step_start_xs"][step_index].to(device=device, dtype=torch.float32)
    cached_output = trace["call_outputs"][skip_call_index - 1].to(device=device, dtype=torch.float32)
    denoiser_calls = 0

    for i in range(step_index, int(sampler.num_steps)):
        t_cur_scalar = steps[i]
        t_next_scalar = steps[i + 1]
        dt = t_next_scalar - t_cur_scalar
        t_cur = t_cur_scalar.repeat(batch_size).to(device=device, dtype=x.dtype)
        t_hat = t_next_scalar.repeat(batch_size).to(device=device, dtype=x.dtype)
        w = sampler.w_scheduler.w(t_cur) if sampler.w_scheduler else 0.0

        predictor_call = 2 * i
        cfg_x = torch.cat([x, x], dim=0)
        cfg_t = t_cur.repeat(2)
        if predictor_call == skip_call_index:
            cfg_out = cached_output.to(device=device, dtype=cfg_x.dtype)
        else:
            cfg_out = net(cfg_x, cfg_t, cfg_condition)
            cached_output = cfg_out.detach()
            denoiser_calls += 1

        v = _guided_velocity(sampler, cfg_out, cfg_x, cfg_t, guidance_t=t_cur)
        s = _score_s(sampler, t_cur, x, v)
        x_hat_state = sampler.step_fn(x, v, dt, s=s, w=w)

        if i < int(sampler.num_steps) - 1:
            corrector_call = 2 * i + 1
            cfg_x_hat = torch.cat([x_hat_state, x_hat_state], dim=0)
            cfg_t_hat = t_hat.repeat(2)
            if corrector_call == skip_call_index:
                cfg_out_hat = cached_output.to(device=device, dtype=cfg_x_hat.dtype)
            else:
                cfg_out_hat = net(cfg_x_hat, cfg_t_hat, cfg_condition)
                cached_output = cfg_out_hat.detach()
                denoiser_calls += 1

            v_hat = _guided_velocity(sampler, cfg_out_hat, cfg_x_hat, cfg_t_hat, guidance_t=t_cur)
            s_hat = _score_s(sampler, t_hat, x_hat_state, v_hat)
            v_avg = (v + v_hat) / 2
            s_avg = (s + s_hat) / 2
            x = sampler.step_fn(x, v_avg, dt, s=s_avg, w=w)
        else:
            x = sampler.last_step_fn(x, v, dt, s=s, w=w)

    return x.detach().cpu().float(), denoiser_calls


@torch.inference_mode()
def dino_features(
    images: torch.Tensor,
    dino_model: Optional[nn.Module],
    device: torch.device,
    args: argparse.Namespace,
) -> Optional[torch.Tensor]:
    if dino_model is None:
        return None

    feats: List[torch.Tensor] = []
    for start in range(0, images.shape[0], args.dino_batch_size):
        batch = images[start : start + args.dino_batch_size].to(device=device, dtype=torch.float32)
        batch = ((batch.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
        if args.dino_size > 0:
            batch = F.interpolate(batch, size=(args.dino_size, args.dino_size), mode="bicubic", align_corners=False)
        batch = E2.normalize_imagenet(batch)
        out = dino_model.forward_features(batch)
        if args.dino_feature == "cls":
            feat = out["x_norm_clstoken"]
        else:
            feat = out["x_norm_patchtokens"].mean(dim=1)
        feats.append(F.normalize(feat.float(), dim=-1).detach().cpu())
    return torch.cat(feats, dim=0)


def dino_distance_from_ref(ref_feat: Optional[torch.Tensor], image: torch.Tensor, dino_model: Optional[nn.Module], device: torch.device, args: argparse.Namespace) -> Optional[float]:
    if ref_feat is None or dino_model is None:
        return None
    cand_feat = dino_features(image, dino_model, device, args)
    if cand_feat is None:
        return None
    return float((1.0 - (ref_feat[:1] * cand_feat[:1]).sum(dim=-1)).item())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def save_bank(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def average_curve_rows(
    *,
    pis_lpips: np.ndarray,
    pis_dino: np.ndarray,
    pis_psnr: np.ndarray,
    tested_mask: np.ndarray,
    valid_mask: np.ndarray,
    call_timesteps: np.ndarray,
    call_steps: np.ndarray,
    call_kinds: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for call_idx in range(call_timesteps.shape[0]):
        rows.append(
            {
                "call_index": call_idx,
                "step": int(call_steps[call_idx]),
                "t": float(call_timesteps[call_idx]),
                "call_kind": call_kind_name(int(call_kinds[call_idx])),
                "selected_for_test": bool(tested_mask[call_idx]),
                "valid_count": int(np.isfinite(pis_lpips[:, call_idx]).sum()),
                "mean_pis_lpips": safe_mean(pis_lpips[:, call_idx]),
                "mean_pis_dino": safe_mean(pis_dino[:, call_idx]),
                "mean_pis_psnr": safe_mean(pis_psnr[:, call_idx]),
                "min_pis_lpips": safe_min(pis_lpips[:, call_idx]),
                "max_pis_lpips": safe_max(pis_lpips[:, call_idx]),
                "valid_any": bool(valid_mask[:, call_idx].any()),
            }
        )
    return rows


def kind_summary_rows(
    pis_lpips: np.ndarray,
    pis_dino: np.ndarray,
    pis_psnr: np.ndarray,
    call_kinds: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for kind in [0, 1]:
        mask = call_kinds == kind
        rows.append(
            {
                "call_kind": call_kind_name(kind),
                "num_calls": int(mask.sum()),
                "valid_lpips_count": int(np.isfinite(pis_lpips[:, mask]).sum()),
                "mean_pis_lpips": safe_mean(pis_lpips[:, mask]),
                "mean_pis_dino": safe_mean(pis_dino[:, mask]),
                "mean_pis_psnr": safe_mean(pis_psnr[:, mask]),
            }
        )
    return rows


def write_lpips_heatmap_csv(path: Path, pis_lpips: np.ndarray) -> None:
    fields = ["sample_index"] + [f"call_{idx:03d}" for idx in range(pis_lpips.shape[1])]
    rows = []
    for sample_idx in range(pis_lpips.shape[0]):
        row: Dict[str, Any] = {"sample_index": sample_idx}
        for call_idx in range(pis_lpips.shape[1]):
            value = pis_lpips[sample_idx, call_idx]
            row[f"call_{call_idx:03d}"] = None if not np.isfinite(value) else float(value)
        rows.append(row)
    write_csv(path, rows, fields)


def write_svg_average_lpips(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    width, height = 920, 340
    pad_l, pad_r, pad_t, pad_b = 58, 24, 28, 44
    values = [
        (int(row["call_index"]), float(row["mean_pis_lpips"]))
        for row in rows
        if row.get("mean_pis_lpips") is not None
    ]
    if not values:
        return
    xs = [item[0] for item in values]
    ys = [item[1] for item in values]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def sx(x: float) -> float:
        return pad_l + (x - x_min) / max(1.0, x_max - x_min) * (width - pad_l - pad_r)

    def sy(y: float) -> float:
        return height - pad_b - (y - y_min) / max(1e-12, y_max - y_min) * (height - pad_t - pad_b)

    points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{pad_l}" y="20" font-family="sans-serif" font-size="14">E5 mean PIS_LPIPS by call</text>',
                f'<line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#333"/>',
                f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" stroke="#333"/>',
                f'<text x="{width / 2:.1f}" y="{height - 8}" text-anchor="middle" font-family="sans-serif" font-size="12">call index</text>',
                f'<text x="14" y="{height / 2:.1f}" transform="rotate(-90 14 {height / 2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="12">mean LPIPS</text>',
                f'<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="2"/>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def write_svg_lpips_heatmap(path: Path, pis_lpips: np.ndarray) -> None:
    finite = pis_lpips[np.isfinite(pis_lpips)]
    if finite.size == 0:
        return
    vmax = float(np.percentile(finite, 95))
    vmax = max(vmax, float(finite.max()), 1e-12)
    cell_w, cell_h = 8, 18
    pad_l, pad_t = 72, 32
    width = pad_l + pis_lpips.shape[1] * cell_w + 24
    height = pad_t + pis_lpips.shape[0] * cell_h + 36

    rects = []
    for sample_idx in range(pis_lpips.shape[0]):
        for call_idx in range(pis_lpips.shape[1]):
            value = pis_lpips[sample_idx, call_idx]
            if not np.isfinite(value):
                fill = "#f1f1f1"
            else:
                alpha = max(0.0, min(1.0, float(value) / vmax))
                red = int(255)
                green = int(245 - 185 * alpha)
                blue = int(235 - 215 * alpha)
                fill = f"#{red:02x}{green:02x}{blue:02x}"
            rects.append(
                f'<rect x="{pad_l + call_idx * cell_w}" y="{pad_t + sample_idx * cell_h}" '
                f'width="{cell_w}" height="{cell_h}" fill="{fill}"/>'
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{pad_l}" y="20" font-family="sans-serif" font-size="14">E5 PIS_LPIPS heatmap</text>',
                *rects,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs_c2i/PixelGen_XL_without_CFG.yaml")
    parser.add_argument("--ckpt", default="ckpts/PixelGen_XL_80ep.ckpt")
    parser.add_argument("--output-dir", default="outputs/e5_pis_bank")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--classes", default=None, help="Comma-separated class ids. Defaults to 0..num_samples-1.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to seed_start..")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--call-stride", type=int, default=2, help="Test every Nth call unless --call-indices is set.")
    parser.add_argument("--call-indices", default=None, help="Optional comma-separated call indices to test.")
    parser.add_argument("--autocast-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--allow-fused-sdpa", action="store_true")
    parser.add_argument("--global-seed", type=int, default=2026)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--dino-repo", default="/root/.cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--dino-model", default="dinov2_vitb14")
    parser.add_argument("--dino-feature", choices=["patch_mean", "cls"], default="patch_mean")
    parser.add_argument("--dino-size", type=int, default=224)
    parser.add_argument("--dino-batch-size", type=int, default=8)
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
        raise RuntimeError("CUDA is not available. E5 requires a GPU-visible shell.")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    E1.configure_runtime(
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
    pairs = E1.class_seed_pairs(args, num_classes=num_classes)
    E1.write_pairs_csv(pairs, output_dir)

    vae = E1.instantiate_from_config(config.model.vae).to(device).eval()
    denoiser = E1.instantiate_from_config(config.model.denoiser).to(device).eval()
    conditioner = E1.instantiate_from_config(config.model.conditioner).to(device).eval()
    sampler = E1.instantiate_from_config(config.model.diffusion_sampler).to(device).eval()
    if not getattr(sampler, "exact_henu", False):
        raise ValueError("E5 single-skip replay currently expects exact_henu=True.")

    prefixes = [item for item in args.weight_prefixes.split(",") if item != ""]
    if args.weight_prefixes.endswith(","):
        prefixes.append("")
    load_info = E1.load_denoiser_weights(denoiser, ckpt_path, prefixes)
    denoiser.to(device).eval()

    lpips_model, lpips_source = E2.load_lpips_model(args, device)
    dino_model, dino_source = E2.load_dino_model(args, device)

    total_calls = E1.total_denoiser_opportunities(sampler)
    call_timesteps, call_steps, call_kinds = build_call_metadata(sampler)
    if total_calls != call_timesteps.shape[0]:
        raise RuntimeError(f"Expected {total_calls} call metadata entries, got {call_timesteps.shape[0]}")

    selected_mask = selected_call_mask(total_calls, args)
    autocast_dtype = torch.bfloat16 if args.autocast_dtype == "bf16" else torch.float16

    num_samples = len(pairs)
    pis_lpips = np.full((num_samples, total_calls), np.nan, dtype=np.float32)
    pis_dino = np.full((num_samples, total_calls), np.nan, dtype=np.float32)
    pis_psnr = np.full((num_samples, total_calls), np.nan, dtype=np.float32)
    pis_ssim = np.full((num_samples, total_calls), np.nan, dtype=np.float32)
    valid_mask = np.zeros((num_samples, total_calls), dtype=bool)
    suffix_denoiser_calls = np.zeros((num_samples, total_calls), dtype=np.int16)

    rows: List[Dict[str, Any]] = []
    preview_images: List[torch.Tensor] = []
    saved_intervention_images = 0
    full_reference_images: List[torch.Tensor] = []

    start_all = time.perf_counter()
    print(
        f"[E5] Running single-skip PIS on {num_samples} samples, "
        f"{int(selected_mask.sum())}/{total_calls} selected calls per sample",
        flush=True,
    )

    for sample_idx, (class_id, seed) in enumerate(pairs):
        sample_start = time.perf_counter()
        print(f"[E5] Sample {sample_idx}/{num_samples - 1}: class={class_id} seed={seed}", flush=True)

        noise = E1.make_noise_batch([(class_id, seed)], latent_shape, device, args.noise_scale)
        condition, uncondition = conditioner([class_id], {})

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=not args.no_autocast):
            trace = run_full_trace(
                net=denoiser,
                sampler=sampler,
                noise=noise,
                condition=condition,
                uncondition=uncondition,
            )
            full_image = vae.decode(trace["final"].to(device)).detach().cpu().float()

        full_reference_images.append(full_image)
        ref_dino_feat = dino_features(full_image, dino_model, device, args)

        if sample_idx < args.save_image_count:
            E1.save_tensor_image(full_image[0], output_dir / "full_reference_images" / f"idx{sample_idx:04d}_class{class_id:04d}_seed{seed}.png")

        for call_idx in range(total_calls):
            row: Dict[str, Any] = {
                "sample_index": sample_idx,
                "class_id": int(class_id),
                "seed": int(seed),
                "call_index": call_idx,
                "t": float(call_timesteps[call_idx]),
                "step": int(call_steps[call_idx]),
                "call_kind": call_kind_name(int(call_kinds[call_idx])),
                "selected_for_test": bool(selected_mask[call_idx]),
                "intervention_valid": False,
                "invalid_reason": None,
                "pis_lpips": None,
                "pis_dino": None,
                "pis_psnr": None,
                "pis_ssim": None,
                "suffix_denoiser_calls": 0,
                "elapsed_sec": None,
            }

            if not selected_mask[call_idx]:
                row["invalid_reason"] = "not_selected"
                rows.append(row)
                continue
            if call_idx == 0:
                row["invalid_reason"] = "no_previous_cache"
                rows.append(row)
                continue

            intervention_start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=not args.no_autocast):
                candidate_latent, denoiser_calls = run_single_skip_suffix(
                    net=denoiser,
                    sampler=sampler,
                    trace=trace,
                    condition=condition,
                    uncondition=uncondition,
                    skip_call_index=call_idx,
                    device=device,
                )
                candidate_image = vae.decode(candidate_latent.to(device)).detach().cpu().float()

            metric = E1.compute_batch_metrics(
                reference=full_image,
                candidate=candidate_image,
                device=device,
                lpips_model=lpips_model,
            )
            dino_distance = dino_distance_from_ref(ref_dino_feat, candidate_image, dino_model, device, args)
            elapsed = time.perf_counter() - intervention_start

            lpips_value = metric["lpips"][0]
            psnr_value = metric["psnr_db"][0]
            ssim_value = metric["ssim"][0]

            pis_lpips[sample_idx, call_idx] = np.nan if lpips_value is None else float(lpips_value)
            pis_dino[sample_idx, call_idx] = np.nan if dino_distance is None else float(dino_distance)
            pis_psnr[sample_idx, call_idx] = float(psnr_value)
            pis_ssim[sample_idx, call_idx] = float(ssim_value)
            valid_mask[sample_idx, call_idx] = True
            suffix_denoiser_calls[sample_idx, call_idx] = int(denoiser_calls)

            row.update(
                {
                    "intervention_valid": True,
                    "invalid_reason": "",
                    "pis_lpips": None if lpips_value is None else float(lpips_value),
                    "pis_dino": None if dino_distance is None else float(dino_distance),
                    "pis_psnr": float(psnr_value),
                    "pis_ssim": float(ssim_value),
                    "suffix_denoiser_calls": int(denoiser_calls),
                    "elapsed_sec": elapsed,
                }
            )
            rows.append(row)

            if saved_intervention_images < args.save_image_count:
                saved_intervention_images += 1
                preview_images.append(candidate_image)
                name = f"idx{sample_idx:04d}_call{call_idx:03d}_{call_kind_name(int(call_kinds[call_idx]))}.png"
                E1.save_tensor_image(candidate_image[0], output_dir / "intervention_images" / name)

        arrays = {
            "pis_lpips": pis_lpips,
            "pis_dino": pis_dino,
            "pis_psnr": pis_psnr,
            "pis_ssim": pis_ssim,
            "valid_mask": valid_mask,
            "selected_call_mask": selected_mask,
            "suffix_denoiser_calls": suffix_denoiser_calls,
            "call_kind": call_kinds,
            "call_step": call_steps,
            "t_values": call_timesteps,
        }
        save_bank(output_dir / "pis_bank_partial.npz", arrays)
        with open(output_dir / "progress.json", "w", encoding="utf-8") as fp:
            json.dump(
                json_ready(
                    {
                        "completed_samples": sample_idx + 1,
                        "total_samples": num_samples,
                        "elapsed_sec": time.perf_counter() - start_all,
                        "last_sample_elapsed_sec": time.perf_counter() - sample_start,
                    }
                ),
                fp,
                indent=2,
            )
        torch.cuda.empty_cache()

    arrays = {
        "pis_lpips": pis_lpips,
        "pis_dino": pis_dino,
        "pis_psnr": pis_psnr,
        "pis_ssim": pis_ssim,
        "valid_mask": valid_mask,
        "selected_call_mask": selected_mask,
        "suffix_denoiser_calls": suffix_denoiser_calls,
        "call_kind": call_kinds,
        "call_step": call_steps,
        "t_values": call_timesteps,
    }
    save_bank(output_dir / "pis_bank.npz", arrays)
    write_csv(output_dir / "pis_summary.csv", rows)

    average_rows = average_curve_rows(
        pis_lpips=pis_lpips,
        pis_dino=pis_dino,
        pis_psnr=pis_psnr,
        tested_mask=selected_mask,
        valid_mask=valid_mask,
        call_timesteps=call_timesteps,
        call_steps=call_steps,
        call_kinds=call_kinds,
    )
    write_csv(output_dir / "pis_average_curves.csv", average_rows)
    write_csv(output_dir / "pis_predictor_corrector_summary.csv", kind_summary_rows(pis_lpips, pis_dino, pis_psnr, call_kinds))
    write_lpips_heatmap_csv(output_dir / "pis_lpips_heatmap.csv", pis_lpips)
    write_svg_average_lpips(output_dir / "pis_mean_lpips.svg", average_rows)
    write_svg_lpips_heatmap(output_dir / "pis_lpips_heatmap.svg", pis_lpips)

    if preview_images:
        E1.save_image_grid(
            torch.cat(preview_images, dim=0),
            output_dir / "intervention_preview_grid.png",
            resize=args.preview_size,
            columns=args.preview_columns,
        )
    if full_reference_images:
        E1.save_image_grid(
            torch.cat(full_reference_images[: args.save_image_count], dim=0),
            output_dir / "full_reference_preview_grid.png",
            resize=args.preview_size,
            columns=args.preview_columns,
        )

    elapsed_all = time.perf_counter() - start_all
    summary = {
        "meta": {
            "config": str(config_path),
            "ckpt": str(ckpt_path),
            "output_dir": str(output_dir),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "num_samples": num_samples,
            "latent_shape": list(latent_shape),
            "sampler": sampler.__class__.__name__,
            "num_steps": int(sampler.num_steps),
            "denoiser_opportunities_per_sample": total_calls,
            "selected_calls_per_sample": int(selected_mask.sum()),
            "call_stride": int(args.call_stride),
            "call_indices": parse_int_list(args.call_indices),
            "allow_fused_sdpa": bool(args.allow_fused_sdpa),
            "no_autocast": bool(args.no_autocast),
            "lpips_source": lpips_source,
            "dino_source": dino_source,
            "dino_feature": args.dino_feature,
            "dino_size": args.dino_size,
            "load_info": load_info,
            "args": vars(args),
        },
        "elapsed_sec": elapsed_all,
        "valid_interventions": int(valid_mask.sum()),
        "pis_bank": str(output_dir / "pis_bank.npz"),
        "pis_summary_csv": str(output_dir / "pis_summary.csv"),
        "average_curves_csv": str(output_dir / "pis_average_curves.csv"),
        "predictor_corrector_summary_csv": str(output_dir / "pis_predictor_corrector_summary.csv"),
        "metric_summary": {
            "pis_lpips": {
                "mean": safe_mean(pis_lpips),
                "min": safe_min(pis_lpips),
                "max": safe_max(pis_lpips),
            },
            "pis_dino": {
                "mean": safe_mean(pis_dino),
                "min": safe_min(pis_dino),
                "max": safe_max(pis_dino),
            },
            "pis_psnr": {
                "mean": safe_mean(pis_psnr),
                "min": safe_min(pis_psnr),
                "max": safe_max(pis_psnr),
            },
            "pis_ssim": {
                "mean": safe_mean(pis_ssim),
                "min": safe_min(pis_ssim),
                "max": safe_max(pis_ssim),
            },
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(json_ready(summary), fp, indent=2)

    print(f"[E5] Summary written to {output_dir / 'summary.json'}", flush=True)
    print(f"[E5] PIS bank written to {output_dir / 'pis_bank.npz'}", flush=True)


if __name__ == "__main__":
    main()
