#!/usr/bin/env python3
"""Dump full no-cache PixelGen trajectories for the E6-D0 offline diagnostic."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
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


E5 = _load_helper_module("07_e5_pis_single_skip.py", "pixelgen_e5_helpers_for_e6d0_dump")
E1 = E5.E1


def resolve_path(path: str | Path) -> Path:
    item = Path(path).expanduser()
    if item.is_absolute():
        return item
    return (PROJECT_ROOT / item).resolve()


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def save_dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported save dtype: {name}")


def tensor_for_save(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    out = tensor.detach().cpu()
    if out.ndim == 4 and out.shape[0] == 1:
        out = out[0]
    return out.to(dtype=dtype).contiguous()


def tensor_bchw(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    out = tensor
    if out.ndim == 3:
        out = out.unsqueeze(0)
    return out.to(device=device, dtype=torch.float32)


def effective_xhat_from_velocity(
    x_input: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    sampler: nn.Module,
) -> torch.Tensor:
    denom = (1.0 - t.view(-1, 1, 1, 1)).clamp_min(float(sampler.t_eps))
    return x_input + denom * v


@torch.inference_mode()
def run_full_trace_with_call_inputs(
    *,
    net: nn.Module,
    sampler: nn.Module,
    noise: torch.Tensor,
    condition: torch.Tensor,
    uncondition: torch.Tensor,
    save_dtype: torch.dtype,
) -> Dict[str, Any]:
    """Run the exact E5/E5.5 full sampler while saving each call input and xhat."""

    batch_size = noise.shape[0]
    if batch_size != 1:
        raise ValueError("E6-D0 trajectory dump expects batch_size=1.")

    steps = sampler.timesteps.to(noise.device)
    cfg_condition = torch.cat([uncondition, condition], dim=0)
    x = noise
    calls: List[Dict[str, Any]] = []
    call_index = 0

    for step_idx, (t_cur_scalar, t_next_scalar) in enumerate(zip(steps[:-1], steps[1:])):
        dt = t_next_scalar - t_cur_scalar
        h_abs = float(abs(float(dt.detach().cpu())))
        t_cur = t_cur_scalar.repeat(batch_size).to(noise.device, noise.dtype)
        t_hat = t_next_scalar.repeat(batch_size).to(noise.device, noise.dtype)
        w = sampler.w_scheduler.w(t_cur) if sampler.w_scheduler else 0.0

        x_input = x
        cfg_out, cfg_x, cfg_t = E5._cfg_denoiser_call(net, x_input, t_cur, cfg_condition)
        v = E5._guided_velocity(sampler, cfg_out, cfg_x, cfg_t, guidance_t=t_cur)
        xhat_eff = effective_xhat_from_velocity(x_input, t_cur, v, sampler)
        s = E5._score_s(sampler, t_cur, x_input, v)
        calls.append(
            {
                "call_index": call_index,
                "step_index": int(step_idx),
                "call_kind": "predictor",
                "t": float(t_cur_scalar.detach().cpu()),
                "t_next": float(t_next_scalar.detach().cpu()),
                "h": h_abs,
                "effective_solver_coeff": h_abs,
                "step_t_start": float(t_cur_scalar.detach().cpu()),
                "step_t_end": float(t_next_scalar.detach().cpu()),
                "x_t": tensor_for_save(x_input, save_dtype),
                "xhat": tensor_for_save(xhat_eff, save_dtype),
            }
        )
        call_index += 1

        x_hat_state = sampler.step_fn(x_input, v, dt, s=s, w=w)

        if step_idx < int(sampler.num_steps) - 1:
            x_corr_input = x_hat_state
            cfg_out_hat, cfg_x_hat, cfg_t_hat = E5._cfg_denoiser_call(net, x_corr_input, t_hat, cfg_condition)
            v_hat = E5._guided_velocity(sampler, cfg_out_hat, cfg_x_hat, cfg_t_hat, guidance_t=t_cur)
            xhat_eff_hat = effective_xhat_from_velocity(x_corr_input, t_hat, v_hat, sampler)
            s_hat = E5._score_s(sampler, t_hat, x_corr_input, v_hat)
            calls.append(
                {
                    "call_index": call_index,
                    "step_index": int(step_idx),
                    "call_kind": "corrector",
                    "t": float(t_next_scalar.detach().cpu()),
                    "t_next": float(t_cur_scalar.detach().cpu()),
                    "h": h_abs,
                    "effective_solver_coeff": h_abs,
                    "step_t_start": float(t_cur_scalar.detach().cpu()),
                    "step_t_end": float(t_next_scalar.detach().cpu()),
                    "x_t": tensor_for_save(x_corr_input, save_dtype),
                    "xhat": tensor_for_save(xhat_eff_hat, save_dtype),
                }
            )
            call_index += 1

            v_avg = (v + v_hat) / 2
            s_avg = (s + s_hat) / 2
            x = sampler.step_fn(x_input, v_avg, dt, s=s_avg, w=w)
        else:
            x = sampler.last_step_fn(x_input, v, dt, s=s, w=w)

    return {"final": x.detach().cpu().float(), "calls": calls, "denoiser_calls": call_index}


@torch.inference_mode()
def replay_saved_calls(
    *,
    calls: Sequence[Mapping[str, Any]],
    sampler: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Replay sampler updates using saved effective xhat tensors only."""

    if not calls:
        raise ValueError("Cannot replay an empty call list")
    x = tensor_bchw(calls[0]["x_t"], device)
    call_idx = 0
    batch_size = 1

    for step_idx in range(int(sampler.num_steps)):
        pred = calls[call_idx]
        if pred["call_kind"] != "predictor":
            raise RuntimeError(f"Expected predictor at call {call_idx}, got {pred['call_kind']}")
        t_cur_value = float(pred["step_t_start"])
        t_next_value = float(pred["step_t_end"])
        t_cur = torch.tensor([t_cur_value], device=device, dtype=x.dtype)
        t_hat = torch.tensor([t_next_value], device=device, dtype=x.dtype)
        dt = torch.tensor(t_next_value - t_cur_value, device=device, dtype=x.dtype)
        w = sampler.w_scheduler.w(t_cur) if sampler.w_scheduler else 0.0

        xhat_pred = tensor_bchw(pred["xhat"], device)
        denom = (1.0 - t_cur.view(batch_size, 1, 1, 1)).clamp_min(float(sampler.t_eps))
        v = (xhat_pred - x) / denom
        s = E5._score_s(sampler, t_cur, x, v)
        x_hat_state = sampler.step_fn(x, v, dt, s=s, w=w)
        call_idx += 1

        if step_idx < int(sampler.num_steps) - 1:
            corr = calls[call_idx]
            if corr["call_kind"] != "corrector":
                raise RuntimeError(f"Expected corrector at call {call_idx}, got {corr['call_kind']}")
            xhat_corr = tensor_bchw(corr["xhat"], device)
            denom_hat = (1.0 - t_hat.view(batch_size, 1, 1, 1)).clamp_min(float(sampler.t_eps))
            v_hat = (xhat_corr - x_hat_state) / denom_hat
            s_hat = E5._score_s(sampler, t_hat, x_hat_state, v_hat)
            v_avg = (v + v_hat) / 2
            s_avg = (s + s_hat) / 2
            x = sampler.step_fn(x, v_avg, dt, s=s_avg, w=w)
            call_idx += 1
        else:
            x = sampler.last_step_fn(x, v, dt, s=s, w=w)

    if call_idx != len(calls):
        raise RuntimeError(f"Replay consumed {call_idx} calls, but trajectory has {len(calls)}")
    return x.detach().cpu().float()


def replay_error(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    ref = reference.detach().cpu().float()
    cand = candidate.detach().cpu().float()
    diff = cand - ref
    mse = float(diff.square().mean().item())
    peak = max(1.0, float(ref.abs().max().item()))
    psnr = float("inf") if mse <= 0.0 else 20.0 * math.log10(peak / math.sqrt(max(mse, 1e-30)))
    return {
        "mean_abs_error": float(diff.abs().mean().item()),
        "max_abs_error": float(diff.abs().max().item()),
        "mse": mse,
        "psnr_db": psnr,
    }


def sample_file_complete(path: Path, expected_calls: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    calls = payload.get("calls")
    meta = payload.get("meta", {})
    return isinstance(calls, list) and len(calls) == expected_calls and int(meta.get("num_calls", -1)) == expected_calls


def write_skipped_summary(out_dir: Path, args: argparse.Namespace, reason: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "skipped",
        "reason": reason,
        "args": vars(args),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / f"summary_shard{args.shard_id}.json", "w", encoding="utf-8") as fp:
        json.dump(json_ready(payload), fp, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs_c2i/PixelGen_XL_without_CFG.yaml")
    parser.add_argument("--ckpt", default="ckpts/PixelGen_XL_80ep.ckpt")
    parser.add_argument("--out", default="outputs/e6_d0_fulltraj/main8")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--classes", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--save-dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--autocast-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--no-autocast", action="store_true", default=True)
    parser.add_argument("--autocast", dest="no_autocast", action="store_false")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--allow-fused-sdpa", action="store_true")
    parser.add_argument("--global-seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-replay", action="store_true")
    parser.add_argument("--preview-count", type=int, default=4)
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
    if args.num_shards <= 0 or args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")

    out_dir = resolve_path(args.out)
    config_path = resolve_path(args.config)
    ckpt_path = resolve_path(args.ckpt)

    if not ckpt_path.exists():
        reason = f"checkpoint missing: {ckpt_path}; skip GPU trajectory dump"
        print(f"[E6-D0 dump] {reason}", flush=True)
        write_skipped_summary(out_dir, args, reason)
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        reason = "CUDA is not available; skip GPU trajectory dump"
        print(f"[E6-D0 dump] {reason}", flush=True)
        write_skipped_summary(out_dir, args, reason)
        return

    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    E1.configure_runtime(
        args.global_seed,
        args.deterministic_algorithms,
        args.allow_tf32,
        force_math_sdpa=not args.allow_fused_sdpa,
    )

    device = torch.device(args.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(config_path)
    num_classes = int(config.model.denoiser.init_args.num_classes)
    latent_shape = config.data.pred_dataset.init_args.latent_shape
    pairs = E1.class_seed_pairs(args, num_classes=num_classes)
    shard_indices = [idx for idx in range(len(pairs)) if idx % int(args.num_shards) == int(args.shard_id)]

    vae = E1.instantiate_from_config(config.model.vae).to(device).eval()
    denoiser = E1.instantiate_from_config(config.model.denoiser).to(device).eval()
    conditioner = E1.instantiate_from_config(config.model.conditioner).to(device).eval()
    sampler = E1.instantiate_from_config(config.model.diffusion_sampler).to(device).eval()
    if not getattr(sampler, "exact_henu", False):
        raise ValueError("E6-D0 dump currently expects exact_henu=True.")

    prefixes = [item for item in args.weight_prefixes.split(",") if item != ""]
    if args.weight_prefixes.endswith(","):
        prefixes.append("")
    load_info = E1.load_denoiser_weights(denoiser, ckpt_path, prefixes)
    denoiser.to(device).eval()

    total_calls = E1.total_denoiser_opportunities(sampler)
    call_timesteps, _call_steps, _call_kinds = E5.build_call_metadata(sampler)
    if total_calls != int(call_timesteps.shape[0]):
        raise RuntimeError(f"Expected {total_calls} call metadata entries, got {call_timesteps.shape[0]}")

    save_dtype = save_dtype_from_name(args.save_dtype)
    autocast_dtype = torch.bfloat16 if args.autocast_dtype == "bf16" else torch.float16
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

    rows: List[Dict[str, Any]] = []
    replay_rows: List[Dict[str, Any]] = []
    preview_images: List[torch.Tensor] = []
    started = time.perf_counter()

    print(
        f"[E6-D0 dump] shard {args.shard_id}/{args.num_shards}: "
        f"{len(shard_indices)} samples, out={out_dir}",
        flush=True,
    )

    for sample_index in shard_indices:
        class_id, seed = pairs[sample_index]
        sample_path = out_dir / f"sample_{sample_index:03d}.pt"
        if args.resume and sample_file_complete(sample_path, total_calls):
            rows.append(
                {
                    "sample_index": sample_index,
                    "sample_id": sample_index,
                    "class_id": int(class_id),
                    "seed": int(seed),
                    "path": str(sample_path),
                    "num_calls": total_calls,
                    "status": "skipped_existing",
                    "elapsed_sec": 0.0,
                }
            )
            print(f"[E6-D0 dump] sample {sample_index}: existing complete file, skip", flush=True)
            continue

        sample_start = time.perf_counter()
        print(f"[E6-D0 dump] sample {sample_index}: class={class_id} seed={seed}", flush=True)
        noise = E1.make_noise_batch([(class_id, seed)], latent_shape, device, args.noise_scale)
        condition, uncondition = conditioner([class_id], {})

        with torch.autocast(device_type=autocast_device_type, dtype=autocast_dtype, enabled=not args.no_autocast):
            trace = run_full_trace_with_call_inputs(
                net=denoiser,
                sampler=sampler,
                noise=noise,
                condition=condition,
                uncondition=uncondition,
                save_dtype=save_dtype,
            )
            final_image = vae.decode(trace["final"].to(device)).detach().cpu().float()

        if int(trace["denoiser_calls"]) != total_calls or len(trace["calls"]) != total_calls:
            raise RuntimeError(f"Expected {total_calls} calls, got {trace['denoiser_calls']}/{len(trace['calls'])}")

        payload = {
            "meta": {
                "sample_id": int(sample_index),
                "sample_index": int(sample_index),
                "class_id": int(class_id),
                "seed": int(seed),
                "config": str(config_path),
                "ckpt": str(ckpt_path),
                "sampler": sampler.__class__.__name__,
                "steps": int(sampler.num_steps),
                "num_calls": int(total_calls),
                "image_size": int(latent_shape[-1]),
                "dtype_saved": "float16" if save_dtype == torch.float16 else "float32",
                "epsilon_clip": float(sampler.t_eps),
                "no_autocast": bool(args.no_autocast),
                "allow_fused_sdpa": bool(args.allow_fused_sdpa),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "final_latent": tensor_for_save(trace["final"], save_dtype),
            "final_image": tensor_for_save(final_image, save_dtype),
            "calls": trace["calls"],
        }
        torch.save(payload, sample_path)

        replay_info: Dict[str, Any] = {}
        if args.check_replay:
            replay_final = replay_saved_calls(calls=trace["calls"], sampler=sampler, device=device)
            replay_info = replay_error(trace["final"], replay_final)
            replay_rows.append(
                {
                    "sample_index": sample_index,
                    "class_id": int(class_id),
                    "seed": int(seed),
                    **replay_info,
                }
            )

        elapsed = time.perf_counter() - sample_start
        row = {
            "sample_index": sample_index,
            "sample_id": sample_index,
            "class_id": int(class_id),
            "seed": int(seed),
            "path": str(sample_path),
            "num_calls": int(total_calls),
            "status": "written",
            "elapsed_sec": elapsed,
            **{f"replay_{key}": value for key, value in replay_info.items()},
        }
        rows.append(row)
        if len(preview_images) < max(0, args.preview_count):
            preview_images.append(final_image)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    index_fields = [
        "sample_index",
        "sample_id",
        "class_id",
        "seed",
        "path",
        "num_calls",
        "status",
        "elapsed_sec",
        "replay_mean_abs_error",
        "replay_max_abs_error",
        "replay_mse",
        "replay_psnr_db",
    ]
    write_csv(out_dir / f"index_shard{args.shard_id}.csv", rows, index_fields)
    if replay_rows:
        write_csv(
            out_dir / f"replay_check_shard{args.shard_id}.csv",
            replay_rows,
            ["sample_index", "class_id", "seed", "mean_abs_error", "max_abs_error", "mse", "psnr_db"],
        )
    if preview_images:
        E1.save_image_grid(
            torch.cat(preview_images, dim=0),
            out_dir / f"preview_grid_shard{args.shard_id}.png",
            resize=args.preview_size,
            columns=args.preview_columns,
        )

    replay_summary: Dict[str, Optional[float]] = {}
    if replay_rows:
        for key in ["mean_abs_error", "max_abs_error", "mse", "psnr_db"]:
            values = np.asarray([float(row[key]) for row in replay_rows], dtype=np.float64)
            replay_summary[key] = float(values.mean()) if np.isfinite(values).all() else None
        replay_summary["max_abs_error_max"] = float(max(row["max_abs_error"] for row in replay_rows))

    summary = {
        "status": "ok",
        "out_dir": str(out_dir),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "num_samples_requested": int(args.num_samples),
        "num_samples_this_shard": int(len(shard_indices)),
        "num_written": int(sum(1 for row in rows if row.get("status") == "written")),
        "num_skipped_existing": int(sum(1 for row in rows if row.get("status") == "skipped_existing")),
        "num_calls": int(total_calls),
        "elapsed_sec": time.perf_counter() - started,
        "save_dtype": args.save_dtype,
        "no_autocast": bool(args.no_autocast),
        "allow_fused_sdpa": bool(args.allow_fused_sdpa),
        "load_info": load_info,
        "replay_summary": replay_summary,
        "args": vars(args),
    }
    with open(out_dir / f"summary_shard{args.shard_id}.json", "w", encoding="utf-8") as fp:
        json.dump(json_ready(summary), fp, indent=2)
    print(f"[E6-D0 dump] shard summary written to {out_dir / f'summary_shard{args.shard_id}.json'}", flush=True)


if __name__ == "__main__":
    main()
