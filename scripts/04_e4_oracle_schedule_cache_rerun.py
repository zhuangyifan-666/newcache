#!/usr/bin/env python3
"""Run E4 fixed oracle-schedule cache reruns for PixelGen.

E4 consumes the matched schedules produced by E3 and reruns real cached
PixelGen inference. Unlike E3, this script does execute the denoiser/cache
loop: if schedule[i, c] is True, call c refreshes the full denoiser output;
otherwise it reuses the cached output from the most recent refresh.

Recommended E4 pilot on the E3 test split:

CUDA_VISIBLE_DEVICES=0 conda run -n pixelgen python scripts/04_e4_oracle_schedule_cache_rerun.py \
  --device cuda:0 \
  --split test \
  --target-rrs 0.30,0.50 \
  --schedule-methods uniform,raw_oracle,sea_oracle,pma_nogate_oracle,pma_stageaware_oracle \
  --no-autocast \
  --run-id e4_test_rr030_rr050_fp32
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_e1_helpers() -> Any:
    path = PROJECT_ROOT / "scripts" / "01_e1_online_cache.py"
    spec = importlib.util.spec_from_file_location("pixelgen_e1_online_cache_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import E1 helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E1 = load_e1_helpers()

from src.diffusion.flow_matching.e1_cache import BaseE1CacheController  # noqa: E402


SCHEDULE_METHODS = (
    "uniform",
    "raw_oracle",
    "sea_oracle",
    "dino_oracle",
    "lpips_oracle",
    "pma_nogate_oracle",
    "pma_stageaware_oracle",
)


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_method_list(value: str) -> List[str]:
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(SCHEDULE_METHODS))
    if unknown:
        raise ValueError(f"Unknown schedule methods: {unknown}. Allowed: {list(SCHEDULE_METHODS)}")
    return methods


def parse_online_method_list(value: str) -> List[str]:
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(methods) - {"raw", "sea"})
    if unknown:
        raise ValueError(f"Unknown online methods: {unknown}. Allowed: ['raw', 'sea']")
    return methods


def rr_label(target_rr: float) -> str:
    return f"rr{target_rr:.2f}".replace(".", "p")


def format_token(value: float) -> str:
    return f"{value:.4g}".replace(".", "p").replace("-", "m")


def scalar_from_np(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return value


def json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return "<tensor omitted>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class FixedScheduleCacheController(BaseE1CacheController):
    """Denoiser-output cache controlled by a precomputed [sample, call] schedule."""

    def __init__(
        self,
        schedule: np.ndarray,
        sample_indices: Sequence[int],
        forced_mask: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        if schedule.ndim != 2:
            raise ValueError(f"Expected schedule with shape [samples, calls], got {schedule.shape}")
        self.schedule = schedule.astype(bool, copy=False)
        self.sample_indices = [int(idx) for idx in sample_indices]
        self.forced_mask = (
            np.asarray(forced_mask, dtype=bool)
            if forced_mask is not None
            else np.zeros((self.schedule.shape[1],), dtype=bool)
        )
        if self.forced_mask.shape[0] != self.schedule.shape[1]:
            raise ValueError("forced_mask length must match schedule call dimension.")
        self.current_sample_index: Optional[int] = None

    def reset(self) -> None:
        super().reset()
        self.current_sample_index = None

    def set_sample_index(self, sample_index: int) -> None:
        sample_index = int(sample_index)
        if sample_index < 0 or sample_index >= self.schedule.shape[0]:
            raise IndexError(f"sample_index {sample_index} is outside schedule rows {self.schedule.shape[0]}")
        self.current_sample_index = sample_index

    def start_sample(self, total_calls: int) -> None:
        if self.current_sample_index is None:
            raise RuntimeError("FixedScheduleCacheController.set_sample_index must be called before sampling.")
        if total_calls != self.schedule.shape[1]:
            raise ValueError(f"Schedule has {self.schedule.shape[1]} calls, sampler expects {total_calls}.")
        super().start_sample(total_calls)

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
        if self.current_sample_index is None:
            raise RuntimeError("No current sample index set.")
        if x.shape[0] != 1:
            raise ValueError("FixedScheduleCacheController currently supports batch_size=1 only.")

        cfg_x = torch.cat([x, x], dim=0)
        cfg_t = t.repeat(2)
        schedule_refresh = bool(self.schedule[self.current_sample_index, call_index])
        forced_refresh = (
            self.state.cached_output is None
            or bool(self.forced_mask[call_index])
            or call_index == 0
        )
        should_refresh = schedule_refresh or forced_refresh

        if should_refresh:
            cfg_output = self._run_denoiser(net, cfg_x, cfg_t, cfg_condition)
            self.state.cached_output = cfg_output.detach()
            self.state.consecutive_hits = 0
            self._record(hit=False, forced_refresh=forced_refresh)
            return cfg_output, cfg_x, cfg_t

        self.state.consecutive_hits += 1
        self._record(hit=True, forced_refresh=False)
        return self.state.cached_output, cfg_x, cfg_t


def collect_schedule_paths(args: argparse.Namespace) -> List[Path]:
    if args.schedule_files:
        return [(PROJECT_ROOT / item.strip()).resolve() for item in args.schedule_files.split(",") if item.strip()]

    schedule_dir = (PROJECT_ROOT / args.schedule_dir).resolve()
    methods = parse_method_list(args.schedule_methods)
    target_rrs = parse_float_list(args.target_rrs)
    paths: List[Path] = []
    for method in methods:
        for target_rr in target_rrs:
            paths.append(schedule_dir / f"{method}_{rr_label(target_rr)}.npz")
    return paths


def load_schedule_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing schedule file: {path}")
    with np.load(path) as data:
        schedule = np.asarray(data["schedule"], dtype=bool)
        return {
            "path": path,
            "schedule": schedule,
            "method": str(scalar_from_np(data["method"])) if "method" in data.files else path.stem.rsplit("_rr", 1)[0],
            "target_rr": float(scalar_from_np(data["target_rr"])) if "target_rr" in data.files else math.nan,
            "selected_delta": float(scalar_from_np(data["selected_delta"])) if "selected_delta" in data.files else math.nan,
            "e3_actual_rr": float(scalar_from_np(data["actual_rr"])) if "actual_rr" in data.files else float(schedule.mean()),
            "call_timesteps": np.asarray(data["call_timesteps"], dtype=np.float32) if "call_timesteps" in data.files else None,
            "call_steps": np.asarray(data["call_steps"], dtype=np.int16) if "call_steps" in data.files else None,
            "call_kinds": np.asarray(data["call_kinds"], dtype=np.int8) if "call_kinds" in data.files else None,
            "calibration_indices": np.asarray(data["calibration_indices"], dtype=np.int64)
            if "calibration_indices" in data.files
            else np.asarray([], dtype=np.int64),
            "test_indices": np.asarray(data["test_indices"], dtype=np.int64)
            if "test_indices" in data.files
            else np.asarray([], dtype=np.int64),
            "completed_sample_indices": np.asarray(data["completed_sample_indices"], dtype=np.int64)
            if "completed_sample_indices" in data.files
            else np.arange(schedule.shape[0], dtype=np.int64),
            "forced_mask": np.asarray(data["forced_mask"], dtype=bool)
            if "forced_mask" in data.files
            else np.zeros((schedule.shape[1],), dtype=bool),
        }


def select_sample_indices(schedule_info: Mapping[str, Any], args: argparse.Namespace) -> np.ndarray:
    schedule = schedule_info["schedule"]
    if args.sample_indices:
        indices = np.asarray(parse_int_list(args.sample_indices), dtype=np.int64)
    elif args.split == "all":
        indices = np.arange(schedule.shape[0], dtype=np.int64)
    elif args.split == "test":
        indices = np.asarray(schedule_info["test_indices"], dtype=np.int64)
    elif args.split == "calibration":
        indices = np.asarray(schedule_info["calibration_indices"], dtype=np.int64)
    else:
        raise ValueError(f"Unknown split: {args.split}")

    if indices.size == 0:
        raise ValueError(f"No sample indices selected for split={args.split}.")
    if args.limit_samples is not None:
        indices = indices[: max(0, int(args.limit_samples))]
    if indices.size == 0:
        raise ValueError("--limit-samples selected zero samples.")
    if int(indices.min()) < 0 or int(indices.max()) >= schedule.shape[0]:
        raise ValueError(f"Selected indices out of range for schedule shape {schedule.shape}.")
    return indices


def make_all_pairs(args: argparse.Namespace, num_classes: int, total_samples: int) -> List[Tuple[int, int]]:
    class _PairArgs:
        pass

    pair_args = _PairArgs()
    pair_args.classes = args.classes
    pair_args.seeds = args.seeds
    pair_args.num_samples = int(args.num_samples or total_samples)
    pair_args.seed_start = args.seed_start
    pairs = E1.class_seed_pairs(pair_args, num_classes=num_classes)
    if len(pairs) < total_samples:
        raise ValueError(f"Need at least {total_samples} class/seed pairs, got {len(pairs)}.")
    return pairs


def rows_to_csv(rows: Sequence[Mapping[str, Any]], path: Path, fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def run_method(
    *,
    name: str,
    kind: str,
    target_rr: Optional[float],
    schedule_path: Optional[Path],
    selected_delta: Optional[float],
    pairs: Sequence[Tuple[int, int]],
    sample_indices: Sequence[int],
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
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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

    print(f"[E4] Running {name} on {len(pairs)} samples", flush=True)
    for local_start, (sample_index, pair) in enumerate(zip(sample_indices, pairs)):
        class_id, seed = pair
        if hasattr(controller, "set_sample_index"):
            controller.set_sample_index(int(sample_index))  # type: ignore[attr-defined]

        noise = E1.make_noise_batch([pair], latent_shape, device, args.noise_scale)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=not args.no_autocast):
            condition, uncondition = conditioner([class_id], {})
            samples = E1.heun_jit_e1_sampling(denoiser, sampler, noise, condition, uncondition, controller)
            decoded = vae.decode(samples)

        decoded_cpu = decoded.detach().cpu().float()
        if return_images:
            final_images.append(decoded_cpu)

        if reference_images is not None:
            ref_cpu = reference_images[local_start : local_start + 1]
            batch_metrics = E1.compute_batch_metrics(ref_cpu, decoded_cpu, device=device, lpips_model=lpips_model)
            metric_rows.append(
                {
                    "index": int(sample_index),
                    "class_id": int(class_id),
                    "seed": int(seed),
                    "psnr_db": batch_metrics["psnr_db"][0],
                    "ssim": batch_metrics["ssim"][0],
                    "lpips": batch_metrics["lpips"][0],
                }
            )

        remaining_to_save = args.save_image_count - preview_image_count
        if remaining_to_save > 0:
            preview_images.append(decoded_cpu)
            preview_image_count += 1
            save_name = f"idx{int(sample_index):04d}_class{class_id:04d}_seed{seed}.png"
            E1.save_tensor_image(decoded_cpu[0], image_dir / save_name)

    elapsed = time.perf_counter() - start
    if preview_images:
        E1.save_image_grid(
            torch.cat(preview_images, dim=0),
            method_dir / "preview_grid.png",
            resize=args.preview_size,
            columns=args.preview_columns,
        )

    per_sample_csv = None
    metric_summary: Dict[str, Any] = {}
    if metric_rows:
        per_sample_csv = method_dir / "per_sample_metrics.csv"
        rows_to_csv(metric_rows, per_sample_csv)

    if metric_rows:
        metric_summary = {
            "psnr_db": E1.summarize_metric(row["psnr_db"] for row in metric_rows),
            "ssim": E1.summarize_metric(row["ssim"] for row in metric_rows),
            "lpips": E1.summarize_metric(row["lpips"] for row in metric_rows),
        }

    final_tensor = torch.cat(final_images, dim=0) if final_images else None
    summary = {
        "name": name,
        "kind": kind,
        "target_rr": target_rr,
        "selected_delta": selected_delta,
        "schedule_path": str(schedule_path) if schedule_path is not None else None,
        "num_samples": len(pairs),
        "elapsed_sec": elapsed,
        "per_sample_csv": str(per_sample_csv) if per_sample_csv else None,
        "method_dir": str(method_dir),
        "metrics": metric_summary,
        **E1.stats_to_summary(controller),
    }
    if final_tensor is not None:
        summary["final_tensor"] = final_tensor
    return summary, metric_rows


def finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            out[int(row["index"])] = value
    return out


def choose_delta(calibration_rows: Sequence[Mapping[str, Any]], target_rr: float) -> Mapping[str, Any]:
    if not calibration_rows:
        raise ValueError("No calibration rows available.")
    return min(
        calibration_rows,
        key=lambda row: (abs(float(row["actual_rr"]) - target_rr), -float(row["actual_rr"])),
    )


def bootstrap_mean_ci(values: Sequence[float], seed: int, num_bootstrap: int) -> Dict[str, Optional[float]]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    if finite.size == 1 or num_bootstrap <= 0:
        mean = float(np.mean(finite))
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": int(finite.size)}
    rng = np.random.default_rng(seed)
    boot = np.empty((num_bootstrap,), dtype=np.float64)
    n = finite.size
    for idx in range(num_bootstrap):
        boot[idx] = float(np.mean(finite[rng.integers(0, n, size=n)]))
    return {
        "mean": float(np.mean(finite)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n": int(n),
    }


def paired_difference_rows(
    metric_rows_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
    method_summaries: Sequence[Mapping[str, Any]],
    baseline_kind: str,
    seed: int,
    num_bootstrap: int,
) -> List[Dict[str, Any]]:
    by_name = {str(item["name"]): item for item in method_summaries}
    baselines: Dict[str, str] = {}
    for item in method_summaries:
        if item.get("kind") != baseline_kind:
            continue
        target_rr = item.get("target_rr")
        if target_rr is None:
            continue
        baselines[rr_label(float(target_rr))] = str(item["name"])

    rows: List[Dict[str, Any]] = []
    for item in method_summaries:
        name = str(item["name"])
        target_rr = item.get("target_rr")
        if name == "full_reference" or target_rr is None:
            continue
        label = rr_label(float(target_rr))
        baseline_name = baselines.get(label)
        if baseline_name is None or baseline_name == name:
            continue
        current_rows = metric_rows_by_name.get(name, [])
        baseline_rows = metric_rows_by_name.get(baseline_name, [])
        current_summary = by_name[name]
        baseline_summary = by_name[baseline_name]

        row: Dict[str, Any] = {
            "method": name,
            "baseline": baseline_name,
            "target_rr": float(target_rr),
            "method_actual_rr": current_summary.get("actual_rr"),
            "baseline_actual_rr": baseline_summary.get("actual_rr"),
            "method_psnr_mean": ((current_summary.get("metrics") or {}).get("psnr_db") or {}).get("mean"),
            "baseline_psnr_mean": ((baseline_summary.get("metrics") or {}).get("psnr_db") or {}).get("mean"),
            "method_lpips_mean": ((current_summary.get("metrics") or {}).get("lpips") or {}).get("mean"),
            "baseline_lpips_mean": ((baseline_summary.get("metrics") or {}).get("lpips") or {}).get("mean"),
        }
        for metric in ["psnr_db", "ssim", "lpips"]:
            current = finite_values(current_rows, metric)
            baseline = finite_values(baseline_rows, metric)
            common = sorted(set(current) & set(baseline))
            diffs = [current[idx] - baseline[idx] for idx in common]
            stats = bootstrap_mean_ci(diffs, seed=seed, num_bootstrap=num_bootstrap)
            row[f"delta_{metric}_mean"] = stats["mean"]
            row[f"delta_{metric}_ci_low"] = stats["ci_low"]
            row[f"delta_{metric}_ci_high"] = stats["ci_high"]
            row[f"delta_{metric}_n"] = stats["n"]
        rows.append(row)
    return rows


def write_method_summary_csv(methods: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = [
        "name",
        "kind",
        "target_rr",
        "selected_delta",
        "schedule_path",
        "e3_schedule_actual_rr_all",
        "online_calibration_actual_rr",
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
                    "selected_delta": item.get("selected_delta"),
                    "schedule_path": item.get("schedule_path"),
                    "e3_schedule_actual_rr_all": item.get("e3_schedule_actual_rr_all"),
                    "online_calibration_actual_rr": item.get("online_calibration_actual_rr"),
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs_c2i/PixelGen_XL_without_CFG.yaml")
    parser.add_argument("--ckpt", default="ckpts/PixelGen_XL_80ep.ckpt")
    parser.add_argument("--output-dir", default="outputs/e4_oracle_schedule_cache")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--classes", default=None, help="Comma-separated class ids. Defaults to idx %% num_classes.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to seed_start..")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--schedule-dir",
        default="outputs/e3_schedule_oracle/e3_main_256_from_e2_fp32_calib64/matched_schedules",
    )
    parser.add_argument("--schedule-files", default=None, help="Comma-separated explicit schedule .npz paths.")
    parser.add_argument(
        "--schedule-methods",
        default="uniform,raw_oracle,sea_oracle,pma_nogate_oracle,pma_stageaware_oracle",
    )
    parser.add_argument("--target-rrs", default="0.30,0.50")
    parser.add_argument("--split", choices=["test", "calibration", "all"], default="test")
    parser.add_argument("--sample-indices", default=None, help="Comma-separated schedule row indices; overrides --split.")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--baseline-kind", default="sea_oracle")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--reference-tensor",
        default=None,
        help="Optional .pt file with reference_images and sample_indices. Skips rerunning full reference.",
    )
    parser.add_argument(
        "--save-reference-tensor",
        default=None,
        help="Optional .pt path to save the full reference tensor for later candidate reruns.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Run or load the full reference, write summary/reference tensor, then exit before cached methods.",
    )
    parser.add_argument(
        "--include-online-baselines",
        action="store_true",
        help="Also run RawInput-online / SEAInput-online baselines on the same selected samples.",
    )
    parser.add_argument("--online-methods", default="raw,sea")
    parser.add_argument("--online-deltas", default="0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.50,0.80")
    parser.add_argument("--online-calib-samples", type=int, default=16)
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


def save_reference_tensor(path: Path, reference_images: torch.Tensor, sample_indices: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "reference_images": reference_images.cpu().float(),
            "sample_indices": sample_indices.astype(np.int64),
            "pairs": [(int(class_id), int(seed)) for class_id, seed in pairs],
        },
        path,
    )


def load_reference_tensor(path: Path, sample_indices: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        reference_images = payload
        stored_indices = None
        stored_pairs = None
    elif isinstance(payload, dict):
        reference_images = payload.get("reference_images")
        stored_indices = payload.get("sample_indices")
        stored_pairs = payload.get("pairs")
    else:
        raise TypeError(f"Unsupported reference tensor payload in {path}: {type(payload)!r}")

    if not isinstance(reference_images, torch.Tensor):
        raise TypeError(f"{path} does not contain a reference_images tensor.")
    if reference_images.shape[0] != len(sample_indices):
        raise ValueError(
            f"Reference tensor sample count {reference_images.shape[0]} does not match selected samples {len(sample_indices)}."
        )
    if stored_indices is not None:
        stored_indices_array = np.asarray(stored_indices, dtype=np.int64)
        if not np.array_equal(stored_indices_array, sample_indices.astype(np.int64)):
            raise ValueError("Reference tensor sample_indices do not match current selected samples.")
    if stored_pairs is not None:
        current_pairs = [(int(class_id), int(seed)) for class_id, seed in pairs]
        stored_pairs_list = [(int(class_id), int(seed)) for class_id, seed in stored_pairs]
        if stored_pairs_list != current_pairs:
            raise ValueError("Reference tensor class/seed pairs do not match current selected samples.")
    return reference_images.cpu().float()


def main() -> None:
    args = build_argparser().parse_args()
    if args.batch_size != 1:
        raise ValueError("E4 fixed per-sample schedules currently require --batch-size 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. E4 requires a GPU-visible shell.")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    schedule_paths = collect_schedule_paths(args)
    schedule_infos = [load_schedule_file(path) for path in schedule_paths]
    if not schedule_infos:
        raise ValueError("No schedule files selected.")

    first_schedule = schedule_infos[0]
    sample_indices = select_sample_indices(first_schedule, args)
    total_schedule_samples, total_calls = first_schedule["schedule"].shape
    for info in schedule_infos[1:]:
        if info["schedule"].shape != (total_schedule_samples, total_calls):
            raise ValueError(f"Schedule shape mismatch: {info['path']} has {info['schedule'].shape}")

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

    config = E1.OmegaConf.load(config_path)
    num_classes = int(config.model.denoiser.init_args.num_classes)
    latent_shape = config.data.pred_dataset.init_args.latent_shape
    all_pairs = make_all_pairs(args, num_classes=num_classes, total_samples=total_schedule_samples)
    pairs = [all_pairs[int(idx)] for idx in sample_indices]
    E1.write_pairs_csv(pairs, output_dir)
    rows_to_csv(
        [
            {
                "local_index": local_idx,
                "schedule_index": int(sample_index),
                "class_id": int(pair[0]),
                "seed": int(pair[1]),
            }
            for local_idx, (sample_index, pair) in enumerate(zip(sample_indices, pairs))
        ],
        output_dir / "selected_class_seed_pairs.csv",
    )

    vae = E1.instantiate_from_config(config.model.vae).to(device).eval()
    denoiser = E1.instantiate_from_config(config.model.denoiser).to(device).eval()
    conditioner = E1.instantiate_from_config(config.model.conditioner).to(device).eval()
    sampler = E1.instantiate_from_config(config.model.diffusion_sampler).to(device).eval()

    expected_calls = E1.total_denoiser_opportunities(sampler)
    if expected_calls != total_calls:
        raise ValueError(f"Schedule call dimension {total_calls} != sampler opportunities {expected_calls}")

    prefixes = [item for item in args.weight_prefixes.split(",") if item != ""]
    if args.weight_prefixes.endswith(","):
        prefixes.append("")
    load_info = E1.load_denoiser_weights(denoiser, ckpt_path, prefixes)
    denoiser.to(device).eval()

    lpips_model, lpips_status = E1.load_lpips_model(args, device)
    if lpips_status:
        print(f"[E4] LPIPS status: {lpips_status}", flush=True)

    method_summaries: List[Dict[str, Any]] = []
    metric_rows_by_name: Dict[str, List[Dict[str, Any]]] = {}
    online_calibration_indices: Optional[np.ndarray] = None

    if args.reference_tensor:
        reference_tensor_path = (PROJECT_ROOT / args.reference_tensor).resolve()
        reference_images = load_reference_tensor(reference_tensor_path, sample_indices, pairs)
        method_summaries.append(
            {
                "name": "full_reference",
                "kind": "full_loaded",
                "target_rr": 1.0,
                "selected_delta": None,
                "schedule_path": None,
                "num_samples": len(pairs),
                "elapsed_sec": 0.0,
                "per_sample_csv": None,
                "method_dir": None,
                "metrics": {},
                "reference_tensor": str(reference_tensor_path),
            }
        )
        print(f"[E4] Loaded reference tensor from {reference_tensor_path}", flush=True)
    else:
        full_controller = E1.AlwaysRefreshController()
        full_summary, _ = run_method(
            name="full_reference",
            kind="full",
            target_rr=1.0,
            schedule_path=None,
            selected_delta=None,
            pairs=pairs,
            sample_indices=sample_indices,
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
        method_summaries.append(full_summary)

    if args.save_reference_tensor:
        reference_tensor_path = (PROJECT_ROOT / args.save_reference_tensor).resolve()
        save_reference_tensor(reference_tensor_path, reference_images, sample_indices, pairs)
        print(f"[E4] Saved reference tensor to {reference_tensor_path}", flush=True)

    if args.reference_only:
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
                "split": args.split,
                "sample_indices": sample_indices.tolist(),
                "num_samples": len(pairs),
                "latent_shape": list(latent_shape),
                "sampler": sampler.__class__.__name__,
                "num_steps": int(sampler.num_steps),
                "denoiser_opportunities_per_sample": expected_calls,
                "reference_tensor": str((PROJECT_ROOT / args.save_reference_tensor).resolve())
                if args.save_reference_tensor
                else args.reference_tensor,
                "args": vars(args),
            },
            "methods": method_summaries,
            "summary_csv": str(summary_csv),
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
            json.dump(json_ready(summary_json), fp, indent=2)
        print(f"[E4] Reference-only summary written to {output_dir / 'summary.json'}", flush=True)
        return

    if args.include_online_baselines:
        online_methods = parse_online_method_list(args.online_methods)
        online_deltas = parse_float_list(args.online_deltas)
        target_rrs = parse_float_list(args.target_rrs)
        online_calibration_indices = np.asarray(first_schedule["calibration_indices"], dtype=np.int64)
        if online_calibration_indices.size == 0:
            online_calibration_indices = sample_indices
        online_calibration_indices = online_calibration_indices[
            : max(1, min(args.online_calib_samples, int(online_calibration_indices.size)))
        ]
        calib_pairs = [all_pairs[int(idx)] for idx in online_calibration_indices]

        for metric in online_methods:
            rows: List[Dict[str, Any]] = []
            print(f"[E4] Calibrating {metric}_online on {len(calib_pairs)} samples", flush=True)
            for delta in online_deltas:
                controller = E1.OnlineInputCacheController(
                    metric=metric,
                    delta=delta,
                    warmup_calls=args.warmup_calls,
                    max_skip_calls=args.max_skip_calls,
                    sea_filter_beta=args.sea_filter_beta,
                    sea_filter_eps=args.sea_filter_eps,
                )
                row = E1.run_sampling_only(
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
                print(f"[E4]   {metric}_online delta={delta:g} -> RR={row['actual_rr']:.4f}", flush=True)

            for target_rr in target_rrs:
                selected = choose_delta(rows, target_rr)
                delta = float(selected["delta"])
                name = f"{metric}_online_{rr_label(target_rr)}_delta{format_token(delta)}"
                controller = E1.OnlineInputCacheController(
                    metric=metric,
                    delta=delta,
                    warmup_calls=args.warmup_calls,
                    max_skip_calls=args.max_skip_calls,
                    sea_filter_beta=args.sea_filter_beta,
                    sea_filter_eps=args.sea_filter_eps,
                )
                summary, metric_rows = run_method(
                    name=name,
                    kind=f"{metric}_online",
                    target_rr=target_rr,
                    schedule_path=None,
                    selected_delta=delta,
                    pairs=pairs,
                    sample_indices=sample_indices,
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
                summary["online_calibration_actual_rr"] = selected["actual_rr"]
                method_summaries.append(summary)
                metric_rows_by_name[name] = metric_rows
                torch.cuda.empty_cache()

    for info in schedule_infos:
        method = str(info["method"])
        target_rr = float(info["target_rr"])
        name = f"{method}_{rr_label(target_rr)}"
        controller = FixedScheduleCacheController(
            schedule=info["schedule"],
            sample_indices=sample_indices,
            forced_mask=info["forced_mask"],
        )
        summary, metric_rows = run_method(
            name=name,
            kind=method,
            target_rr=target_rr,
            schedule_path=Path(info["path"]),
            selected_delta=float(info["selected_delta"]) if math.isfinite(float(info["selected_delta"])) else None,
            pairs=pairs,
            sample_indices=sample_indices,
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
        summary["e3_schedule_actual_rr_all"] = info["e3_actual_rr"]
        method_summaries.append(summary)
        metric_rows_by_name[name] = metric_rows
        torch.cuda.empty_cache()

    summary_csv = output_dir / "method_summary.csv"
    write_method_summary_csv(method_summaries, summary_csv)

    pairwise_rows = paired_difference_rows(
        metric_rows_by_name,
        method_summaries,
        baseline_kind=args.baseline_kind,
        seed=args.global_seed,
        num_bootstrap=args.bootstrap_samples,
    )
    pairwise_csv = output_dir / "paired_delta_vs_baseline.csv"
    pairwise_fields = [
        "method",
        "baseline",
        "target_rr",
        "method_actual_rr",
        "baseline_actual_rr",
        "method_psnr_mean",
        "baseline_psnr_mean",
        "method_lpips_mean",
        "baseline_lpips_mean",
        "delta_psnr_db_mean",
        "delta_psnr_db_ci_low",
        "delta_psnr_db_ci_high",
        "delta_psnr_db_n",
        "delta_ssim_mean",
        "delta_ssim_ci_low",
        "delta_ssim_ci_high",
        "delta_ssim_n",
        "delta_lpips_mean",
        "delta_lpips_ci_low",
        "delta_lpips_ci_high",
        "delta_lpips_n",
    ]
    if pairwise_rows:
        rows_to_csv(pairwise_rows, pairwise_csv, pairwise_fields)

    summary_json = {
        "meta": {
            "config": str(config_path),
            "ckpt": str(ckpt_path),
            "output_dir": str(output_dir),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "split": args.split,
            "sample_indices": sample_indices.tolist(),
            "online_calibration_indices": online_calibration_indices.tolist()
            if online_calibration_indices is not None
            else None,
            "num_samples": len(pairs),
            "latent_shape": list(latent_shape),
            "sampler": sampler.__class__.__name__,
            "num_steps": int(sampler.num_steps),
            "denoiser_opportunities_per_sample": expected_calls,
            "allow_fused_sdpa": bool(args.allow_fused_sdpa),
            "no_autocast": bool(args.no_autocast),
            "lpips_status": lpips_status,
            "load_info": load_info,
            "args": vars(args),
        },
        "schedules": [
            {
                "path": str(info["path"]),
                "method": info["method"],
                "target_rr": info["target_rr"],
                "selected_delta": info["selected_delta"],
                "e3_actual_rr_all": info["e3_actual_rr"],
            }
            for info in schedule_infos
        ],
        "methods": method_summaries,
        "summary_csv": str(summary_csv),
        "pairwise_csv": str(pairwise_csv) if pairwise_rows else None,
        "pairwise": pairwise_rows,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(json_ready(summary_json), fp, indent=2)

    print(f"[E4] Summary written to {output_dir / 'summary.json'}", flush=True)
    print(f"[E4] Method table written to {summary_csv}", flush=True)
    if pairwise_rows:
        print(f"[E4] Paired deltas written to {pairwise_csv}", flush=True)


if __name__ == "__main__":
    main()
