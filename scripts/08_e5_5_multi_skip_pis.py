#!/usr/bin/env python3
"""E5.5 continuous multi-skip Perceptual Intervention Score for PixelGen.

E5 measured the final-image damage caused by skipping exactly one denoiser
call. E5.5 asks the cache question that E5 intentionally avoided: if a real
cache reuses the same output for several consecutive calls, does the error stay
small or accumulate?

The intervention for a window (s, L) is:

    calls < s:        full compute
    calls s..s+L-1:   reuse the last refreshed denoiser output
    calls > s+L-1:    full compute again

This script mirrors scripts/07_e5_pis_single_skip.py and keeps the same Heun
call-level replay logic. The only semantic change is that one skip call becomes
a skip set.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


E5 = _load_helper_module("07_e5_pis_single_skip.py", "pixelgen_e5_helpers")
E1 = E5.E1
E2 = E5.E2


WINDOW_FIELDS = ["window_id", "stage", "start_call", "window_len", "end_call", "note"]

SUMMARY_FIELDS = [
    "sample_index",
    "class_id",
    "seed",
    "window_id",
    "stage",
    "start_call",
    "end_call",
    "window_len",
    "start_step",
    "end_step",
    "start_t",
    "end_t",
    "start_call_kind",
    "end_call_kind",
    "num_predictor",
    "num_corrector",
    "intervention_valid",
    "invalid_reason",
    "pis_lpips",
    "pis_dino",
    "pis_psnr",
    "pis_ssim",
    "suffix_denoiser_calls",
    "skipped_calls",
    "elapsed_sec",
    "reference_image",
    "intervention_image",
]

AGGREGATE_FIELDS = [
    "window_id",
    "stage",
    "start_call",
    "end_call",
    "window_len",
    "valid_count",
    "mean_lpips",
    "median_lpips",
    "q90_lpips",
    "q95_lpips",
    "max_lpips",
    "mean_dino",
    "median_dino",
    "q90_dino",
    "q95_dino",
    "max_dino",
    "mean_psnr",
    "q05_psnr",
    "mean_ssim",
    "q05_ssim",
    "failure_rate_lpips_gt_0p001",
    "failure_rate_lpips_gt_0p005",
    "failure_rate_lpips_gt_0p01",
    "failure_rate_lpips_gt_0p02",
    "failure_rate_dino_gt_0p001",
    "failure_rate_dino_gt_0p005",
    "failure_rate_dino_gt_0p01",
    "failure_rate_dino_gt_0p02",
    "status",
]


def parse_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float):
        return value
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return float("nan")
    return float(text)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_values(rows: Iterable[Mapping[str, Any]], key: str) -> np.ndarray:
    values = [parse_float(row.get(key)) for row in rows]
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def stat_mean(values: np.ndarray) -> Optional[float]:
    return None if values.size == 0 else float(values.mean())


def stat_quantile(values: np.ndarray, q: float) -> Optional[float]:
    return None if values.size == 0 else float(np.quantile(values, q))


def stat_max(values: np.ndarray) -> Optional[float]:
    return None if values.size == 0 else float(values.max())


def finite_or_none(value: Any) -> Optional[float]:
    item = parse_float(value)
    return float(item) if math.isfinite(item) else None


def _add_window(
    rows: List[Dict[str, Any]],
    *,
    stage: str,
    start_call: int,
    window_len: int,
    total_calls: int,
    note: str = "",
) -> None:
    end_call = int(start_call) + int(window_len) - 1
    if start_call <= 0:
        return
    if window_len <= 0:
        return
    if end_call >= total_calls:
        return
    rows.append(
        {
            "window_id": len(rows),
            "stage": stage,
            "start_call": int(start_call),
            "window_len": int(window_len),
            "end_call": int(end_call),
            "note": note,
        }
    )


def generate_windows(preset: str, total_calls: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(stage: str, start: int, length: int, note: str = "") -> None:
        _add_window(rows, stage=stage, start_call=start, window_len=length, total_calls=total_calls, note=note)

    if preset == "smoke":
        for start, length, note in [
            (1, 1, "match_e5_call1"),
            (3, 1, "match_e5_call3"),
            (7, 1, "match_e5_call7"),
            (16, 1, "match_e5_call16"),
            (64, 1, "match_e5_call64"),
            (96, 1, "match_e5_call96"),
            (8, 2, "transition_len2"),
            (16, 2, "mid_len2"),
            (16, 4, "mid_len4"),
            (32, 4, "mid_len4"),
            (64, 8, "mid_len8"),
            (90, 2, "tail_len2"),
            (94, 4, "tail_len4"),
        ]:
            if start < 8:
                stage = "sanity" if length == 1 else "early"
            elif start < 16:
                stage = "sanity" if length == 1 else "transition"
            elif start < 88:
                stage = "sanity" if length == 1 else "mid"
            else:
                stage = "sanity" if length == 1 else "tail"
            add(stage, start, length, note)
        return rows

    if preset == "minimal":
        for start in [1, 3, 5, 7, 16, 48, 80, 96]:
            add("sanity", start, 1, f"match_e5_call{start}")
        for start in [1, 3, 5, 7]:
            add("early", start, 2, "early_len2")
        for start in [16, 32, 48, 64, 80]:
            for length in [2, 4, 8]:
                add("mid", start, length, f"mid_len{length}")
        for start, length in [(90, 2), (90, 4), (94, 2), (94, 4), (96, 2)]:
            add("tail", start, length, f"tail_len{length}")
        return rows

    if preset in {"main8", "full"}:
        for start in [1, 2, 3, 5, 7, 8, 16, 32, 64, 90, 98]:
            add("sanity", start, 1, f"match_e5_call{start}")
        for start in [1, 3, 5, 7]:
            add("early", start, 2, "early_len2")
        for start in [1, 3]:
            add("early", start, 4, "early_len4")
        for start in [8, 10, 12, 14]:
            for length in [2, 4]:
                add("transition", start, length, f"transition_len{length}")
        mid_specs = {
            16: [2, 4, 8, 12, 16],
            20: [2, 4, 8],
            24: [2, 4, 8],
            32: [2, 4, 8, 12],
            40: [2, 4, 8, 16],
            50: [2, 4, 8, 12],
            60: [2, 4, 8],
            70: [2, 4, 8, 12],
            80: [2, 4, 8],
        }
        for start, lengths in mid_specs.items():
            for length in lengths:
                add("mid", start, length, f"mid_len{length}")
        for start in [88, 90, 92, 94, 96]:
            for length in [2, 4, 6]:
                add("tail", start, length, f"tail_len{length}")
        return rows

    if preset == "expanded":
        for start in [1, 3, 7, 16, 64, 98]:
            add("sanity", start, 1, f"match_e5_call{start}")
        for start, length in [(1, 2), (3, 2), (5, 2), (1, 4)]:
            add("early", start, length, f"early_len{length}")
        for start in [8, 10, 12, 14]:
            for length in [2, 4]:
                add("transition", start, length, f"transition_len{length}")
        for start in [16, 32, 50, 70, 80]:
            for length in [2, 4, 8]:
                add("mid", start, length, f"mid_len{length}")
        for start in [90, 94, 96]:
            for length in [2, 4]:
                add("tail", start, length, f"tail_len{length}")
        return rows

    raise ValueError(f"Unknown windows preset: {preset}")


def load_windows_csv(path: Path, total_calls: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader):
            start = int(row["start_call"])
            length = int(row["window_len"])
            end = int(row.get("end_call") or (start + length - 1))
            if end != start + length - 1:
                raise ValueError(f"Invalid window at input row {idx}: end_call does not match start+len-1")
            if start <= 0:
                raise ValueError(f"Invalid window at input row {idx}: start_call must be > 0")
            if length <= 0:
                raise ValueError(f"Invalid window at input row {idx}: window_len must be > 0")
            if end >= total_calls:
                raise ValueError(f"Invalid window at input row {idx}: end_call {end} >= total_calls {total_calls}")
            rows.append(
                {
                    "window_id": idx,
                    "stage": row.get("stage") or infer_stage(start),
                    "start_call": start,
                    "window_len": length,
                    "end_call": end,
                    "note": row.get("note", ""),
                }
            )
    return rows


def infer_stage(start_call: int) -> str:
    if start_call <= 7:
        return "early"
    if start_call <= 15:
        return "transition"
    if start_call <= 87:
        return "mid"
    return "tail"


def shard_windows(windows: Sequence[Mapping[str, Any]], shard_id: int, num_shards: int) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return [dict(row) for row in windows]
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    return [dict(row) for row in windows if int(row["window_id"]) % num_shards == shard_id]


def write_windows_csv(path: Path, windows: Sequence[Mapping[str, Any]]) -> None:
    E5.write_csv(path, windows, WINDOW_FIELDS)


def window_metadata(
    window: Mapping[str, Any],
    call_timesteps: np.ndarray,
    call_steps: np.ndarray,
    call_kinds: np.ndarray,
) -> Dict[str, Any]:
    start = int(window["start_call"])
    end = int(window["end_call"])
    kinds = call_kinds[start : end + 1]
    return {
        "start_step": int(call_steps[start]),
        "end_step": int(call_steps[end]),
        "start_t": float(call_timesteps[start]),
        "end_t": float(call_timesteps[end]),
        "start_call_kind": E5.call_kind_name(int(call_kinds[start])),
        "end_call_kind": E5.call_kind_name(int(call_kinds[end])),
        "num_predictor": int((kinds == 0).sum()),
        "num_corrector": int((kinds == 1).sum()),
    }


@torch.inference_mode()
def run_multi_skip_suffix(
    *,
    net: nn.Module,
    sampler: nn.Module,
    trace: Mapping[str, Any],
    condition: torch.Tensor,
    uncondition: torch.Tensor,
    start_call: int,
    window_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int, int]:
    if start_call <= 0:
        raise ValueError("call 0 cannot be skipped because no cached output exists yet.")

    total_calls = len(trace["call_outputs"])
    end_call = start_call + window_len - 1
    if end_call >= total_calls:
        raise ValueError(f"skip window [{start_call}, {end_call}] outside total calls {total_calls}")

    skip_calls = set(range(start_call, end_call + 1))
    step_index = start_call // 2
    batch_size = 1
    steps = sampler.timesteps.to(device)
    cfg_condition = torch.cat([uncondition, condition], dim=0)
    x = trace["step_start_xs"][step_index].to(device=device, dtype=torch.float32)
    cached_output = trace["call_outputs"][start_call - 1].to(device=device, dtype=torch.float32)
    denoiser_calls = 0
    skipped_calls = 0

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
        if predictor_call in skip_calls:
            cfg_out = cached_output.to(device=device, dtype=cfg_x.dtype)
            skipped_calls += 1
        else:
            cfg_out = net(cfg_x, cfg_t, cfg_condition)
            cached_output = cfg_out.detach()
            denoiser_calls += 1

        v = E5._guided_velocity(sampler, cfg_out, cfg_x, cfg_t, guidance_t=t_cur)
        s = E5._score_s(sampler, t_cur, x, v)
        x_hat_state = sampler.step_fn(x, v, dt, s=s, w=w)

        if i < int(sampler.num_steps) - 1:
            corrector_call = 2 * i + 1
            cfg_x_hat = torch.cat([x_hat_state, x_hat_state], dim=0)
            cfg_t_hat = t_hat.repeat(2)
            if corrector_call in skip_calls:
                cfg_out_hat = cached_output.to(device=device, dtype=cfg_x_hat.dtype)
                skipped_calls += 1
            else:
                cfg_out_hat = net(cfg_x_hat, cfg_t_hat, cfg_condition)
                cached_output = cfg_out_hat.detach()
                denoiser_calls += 1

            v_hat = E5._guided_velocity(sampler, cfg_out_hat, cfg_x_hat, cfg_t_hat, guidance_t=t_cur)
            s_hat = E5._score_s(sampler, t_hat, x_hat_state, v_hat)
            v_avg = (v + v_hat) / 2
            s_avg = (s + s_hat) / 2
            x = sampler.step_fn(x, v_avg, dt, s=s_avg, w=w)
        else:
            x = sampler.last_step_fn(x, v, dt, s=s, w=w)

    if skipped_calls != window_len:
        raise RuntimeError(f"Expected {window_len} skipped calls, got {skipped_calls}")
    return x.detach().cpu().float(), denoiser_calls, skipped_calls


def init_bank_arrays(num_samples: int, num_windows: int) -> Dict[str, np.ndarray]:
    return {
        "pis_lpips": np.full((num_samples, num_windows), np.nan, dtype=np.float32),
        "pis_dino": np.full((num_samples, num_windows), np.nan, dtype=np.float32),
        "pis_psnr": np.full((num_samples, num_windows), np.nan, dtype=np.float32),
        "pis_ssim": np.full((num_samples, num_windows), np.nan, dtype=np.float32),
        "valid_mask": np.zeros((num_samples, num_windows), dtype=bool),
        "suffix_denoiser_calls": np.zeros((num_samples, num_windows), dtype=np.int16),
        "skipped_calls": np.zeros((num_samples, num_windows), dtype=np.int16),
    }


def fill_arrays_from_rows(arrays: Dict[str, np.ndarray], rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        if not parse_bool(row.get("intervention_valid")):
            continue
        sample_idx = int(row["sample_index"])
        window_id = int(row["window_id"])
        arrays["pis_lpips"][sample_idx, window_id] = parse_float(row.get("pis_lpips"))
        arrays["pis_dino"][sample_idx, window_id] = parse_float(row.get("pis_dino"))
        arrays["pis_psnr"][sample_idx, window_id] = parse_float(row.get("pis_psnr"))
        arrays["pis_ssim"][sample_idx, window_id] = parse_float(row.get("pis_ssim"))
        arrays["valid_mask"][sample_idx, window_id] = True
        arrays["suffix_denoiser_calls"][sample_idx, window_id] = int(float(row.get("suffix_denoiser_calls") or 0))
        arrays["skipped_calls"][sample_idx, window_id] = int(float(row.get("skipped_calls") or 0))


def read_existing_summary(path: Path) -> Tuple[List[Dict[str, Any]], set[Tuple[int, int]]]:
    if not path.exists():
        return [], set()
    with open(path, "r", newline="", encoding="utf-8") as fp:
        rows = [dict(row) for row in csv.DictReader(fp)]
    completed = {
        (int(row["sample_index"]), int(row["window_id"]))
        for row in rows
        if parse_bool(row.get("intervention_valid"))
    }
    return rows, completed


def metric_summary(arr: np.ndarray) -> Dict[str, Optional[float]]:
    return {
        "mean": E5.safe_mean(arr),
        "min": E5.safe_min(arr),
        "max": E5.safe_max(arr),
    }


def classify_window(row: Mapping[str, Any]) -> str:
    mean_lpips = finite_or_none(row.get("mean_lpips"))
    q95_lpips = finite_or_none(row.get("q95_lpips"))
    mean_dino = finite_or_none(row.get("mean_dino"))
    q95_dino = finite_or_none(row.get("q95_dino"))
    if mean_lpips is None or q95_lpips is None or mean_dino is None or q95_dino is None:
        return "unknown"
    if mean_lpips > 0.01 or q95_lpips > 0.02 or mean_dino > 0.01 or q95_dino > 0.02:
        return "dangerous"
    if mean_lpips <= 0.002 and q95_lpips <= 0.005 and mean_dino <= 0.002 and q95_dino <= 0.005:
        return "safe"
    return "borderline"


def aggregate_window_rows(
    summary_rows: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_window: Dict[int, List[Mapping[str, Any]]] = {int(row["window_id"]): [] for row in windows}
    for row in summary_rows:
        if parse_bool(row.get("intervention_valid")):
            by_window.setdefault(int(row["window_id"]), []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for window in windows:
        window_id = int(window["window_id"])
        rows = by_window.get(window_id, [])
        lpips = finite_values(rows, "pis_lpips")
        dino = finite_values(rows, "pis_dino")
        psnr = finite_values(rows, "pis_psnr")
        ssim = finite_values(rows, "pis_ssim")

        out: Dict[str, Any] = {
            "window_id": window_id,
            "stage": window["stage"],
            "start_call": int(window["start_call"]),
            "end_call": int(window["end_call"]),
            "window_len": int(window["window_len"]),
            "valid_count": int(len(rows)),
            "mean_lpips": stat_mean(lpips),
            "median_lpips": stat_quantile(lpips, 0.50),
            "q90_lpips": stat_quantile(lpips, 0.90),
            "q95_lpips": stat_quantile(lpips, 0.95),
            "max_lpips": stat_max(lpips),
            "mean_dino": stat_mean(dino),
            "median_dino": stat_quantile(dino, 0.50),
            "q90_dino": stat_quantile(dino, 0.90),
            "q95_dino": stat_quantile(dino, 0.95),
            "max_dino": stat_max(dino),
            "mean_psnr": stat_mean(psnr),
            "q05_psnr": stat_quantile(psnr, 0.05),
            "mean_ssim": stat_mean(ssim),
            "q05_ssim": stat_quantile(ssim, 0.05),
        }
        for threshold, label in [(0.001, "0p001"), (0.005, "0p005"), (0.01, "0p01"), (0.02, "0p02")]:
            out[f"failure_rate_lpips_gt_{label}"] = None if lpips.size == 0 else float((lpips > threshold).mean())
            out[f"failure_rate_dino_gt_{label}"] = None if dino.size == 0 else float((dino > threshold).mean())
        out["status"] = classify_window(out)
        aggregate_rows.append(out)
    return aggregate_rows


def stage_safe_age_rows(aggregate_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    stages = []
    for row in aggregate_rows:
        stage = str(row["stage"])
        if stage not in stages:
            stages.append(stage)

    out_rows: List[Dict[str, Any]] = []
    for stage in stages:
        rows = [row for row in aggregate_rows if str(row["stage"]) == stage and int(row.get("valid_count") or 0) > 0]
        if not rows:
            continue
        starts = [int(row["start_call"]) for row in rows]
        ends = [int(row["end_call"]) for row in rows]
        lengths = sorted({int(row["window_len"]) for row in rows})

        safe_len_mean = 0
        safe_len_q95 = 0
        for length in lengths:
            length_rows = [row for row in rows if int(row["window_len"]) == length]
            mean_safe = all(
                (finite_or_none(row.get("mean_lpips")) is not None)
                and (finite_or_none(row.get("mean_dino")) is not None)
                and float(row["mean_lpips"]) <= 0.002
                and float(row["mean_dino"]) <= 0.002
                for row in length_rows
            )
            q95_safe = all(classify_window(row) == "safe" for row in length_rows)
            if mean_safe:
                safe_len_mean = max(safe_len_mean, length)
            if q95_safe:
                safe_len_q95 = max(safe_len_q95, length)

        if stage == "sanity":
            recommended = None
            reason = "L=1 sanity rows are for E5 reproduction, not an E6 stage policy."
        elif safe_len_q95 <= 0:
            recommended = 0
            reason = "No tested length passes the conservative mean/q95 LPIPS+DINO safety rule."
        else:
            recommended = max(1, safe_len_q95 // 2)
            reason = "Conservative half of the largest fully safe tested window length."

        out_rows.append(
            {
                "stage": stage,
                "start_range": f"{min(starts)}-{max(ends)}",
                "safe_len_mean": safe_len_mean,
                "safe_len_q95": safe_len_q95,
                "recommended_max_age_for_e6": recommended,
                "reason": reason,
            }
        )
    return out_rows


def load_e5_summary(path: Path) -> Dict[Tuple[int, int], Dict[str, float]]:
    if not path.exists():
        return {}
    values: Dict[Tuple[int, int], Dict[str, float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if not parse_bool(row.get("intervention_valid")):
                continue
            sample_idx = int(row["sample_index"])
            call_idx = int(row["call_index"])
            values[(sample_idx, call_idx)] = {
                "lpips": parse_float(row.get("pis_lpips")),
                "dino": parse_float(row.get("pis_dino")),
            }
    return values


def write_synergy_with_e5(
    path: Path,
    l1_path: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    e5_values: Mapping[Tuple[int, int], Mapping[str, float]],
) -> Tuple[int, int]:
    if not e5_values:
        E5.write_csv(
            path,
            [],
            [
                "sample_index",
                "window_id",
                "start_call",
                "end_call",
                "window_len",
                "available_single_count",
                "single_sum_lpips",
                "window_lpips",
                "ratio_lpips",
                "single_sum_dino",
                "window_dino",
                "ratio_dino",
            ],
        )
        E5.write_csv(
            l1_path,
            [],
            [
                "sample_index",
                "window_id",
                "call_index",
                "e55_lpips",
                "e5_lpips",
                "abs_diff_lpips",
                "e55_dino",
                "e5_dino",
                "abs_diff_dino",
            ],
        )
        return 0, 0

    rows: List[Dict[str, Any]] = []
    l1_rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        if not parse_bool(row.get("intervention_valid")):
            continue
        sample_idx = int(row["sample_index"])
        start = int(row["start_call"])
        end = int(row["end_call"])
        singles = [e5_values.get((sample_idx, call_idx)) for call_idx in range(start, end + 1)]
        available = [item for item in singles if item is not None]
        if not available:
            continue
        single_lpips = np.asarray([item["lpips"] for item in available], dtype=np.float64)
        single_dino = np.asarray([item["dino"] for item in available], dtype=np.float64)
        single_sum_lpips = float(np.nansum(single_lpips))
        single_sum_dino = float(np.nansum(single_dino))
        window_lpips = parse_float(row.get("pis_lpips"))
        window_dino = parse_float(row.get("pis_dino"))
        out = {
            "sample_index": sample_idx,
            "window_id": int(row["window_id"]),
            "start_call": start,
            "end_call": end,
            "window_len": int(row["window_len"]),
            "available_single_count": len(available),
            "single_sum_lpips": single_sum_lpips,
            "window_lpips": window_lpips,
            "ratio_lpips": window_lpips / (single_sum_lpips + 1e-8),
            "single_sum_dino": single_sum_dino,
            "window_dino": window_dino,
            "ratio_dino": window_dino / (single_sum_dino + 1e-8),
        }
        rows.append(out)
        if int(row["window_len"]) == 1 and len(available) == 1:
            l1_rows.append(
                {
                    "sample_index": sample_idx,
                    "window_id": int(row["window_id"]),
                    "call_index": start,
                    "e55_lpips": window_lpips,
                    "e5_lpips": single_sum_lpips,
                    "abs_diff_lpips": abs(window_lpips - single_sum_lpips),
                    "e55_dino": window_dino,
                    "e5_dino": single_sum_dino,
                    "abs_diff_dino": abs(window_dino - single_sum_dino),
                }
            )

    E5.write_csv(
        path,
        rows,
        [
            "sample_index",
            "window_id",
            "start_call",
            "end_call",
            "window_len",
            "available_single_count",
            "single_sum_lpips",
            "window_lpips",
            "ratio_lpips",
            "single_sum_dino",
            "window_dino",
            "ratio_dino",
        ],
    )
    E5.write_csv(
        l1_path,
        l1_rows,
        [
            "sample_index",
            "window_id",
            "call_index",
            "e55_lpips",
            "e5_lpips",
            "abs_diff_lpips",
            "e55_dino",
            "e5_dino",
            "abs_diff_dino",
        ],
    )
    return len(rows), len(l1_rows)


def write_heatmap_png(
    path: Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    title: str,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plot_rows = [
        row
        for row in aggregate_rows
        if str(row.get("stage")) != "sanity" and finite_or_none(row.get(value_key)) is not None
    ]
    if not plot_rows:
        return False
    starts = sorted({int(row["start_call"]) for row in plot_rows})
    lengths = sorted({int(row["window_len"]) for row in plot_rows})
    start_index = {value: idx for idx, value in enumerate(starts)}
    len_index = {value: idx for idx, value in enumerate(lengths)}
    matrix = np.full((len(lengths), len(starts)), np.nan, dtype=np.float64)
    for row in plot_rows:
        matrix[len_index[int(row["window_len"])], start_index[int(row["start_call"])]] = parse_float(row[value_key])

    path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(8.0, len(starts) * 0.35)
    fig_h = max(3.5, len(lengths) * 0.45)
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label=value_key)
    plt.yticks(range(len(lengths)), lengths)
    plt.xticks(range(len(starts)), starts, rotation=90)
    plt.xlabel("start_call")
    plt.ylabel("window_len")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return True


def update_top_failure(
    bucket: List[Dict[str, Any]],
    *,
    row: Mapping[str, Any],
    full_image: torch.Tensor,
    candidate_image: torch.Tensor,
    limit: int,
) -> None:
    lpips = finite_or_none(row.get("pis_lpips"))
    if lpips is None:
        return
    bucket.append({"score": lpips, "row": dict(row), "images": torch.cat([full_image, candidate_image], dim=0).cpu()})
    bucket.sort(key=lambda item: float(item["score"]), reverse=True)
    del bucket[limit:]


def update_safe_long(
    bucket: List[Dict[str, Any]],
    *,
    row: Mapping[str, Any],
    full_image: torch.Tensor,
    candidate_image: torch.Tensor,
    limit: int,
    min_len: int,
) -> None:
    length = int(row["window_len"])
    lpips = finite_or_none(row.get("pis_lpips"))
    dino = finite_or_none(row.get("pis_dino"))
    if lpips is None or dino is None:
        return
    if length < min_len or lpips > 0.002 or dino > 0.002:
        return
    score = (length, -lpips, -dino)
    bucket.append({"score": score, "row": dict(row), "images": torch.cat([full_image, candidate_image], dim=0).cpu()})
    bucket.sort(key=lambda item: item["score"], reverse=True)
    del bucket[limit:]


def write_visual_bucket(
    bucket: Sequence[Mapping[str, Any]],
    *,
    grid_path: Path,
    csv_path: Path,
    args: argparse.Namespace,
) -> None:
    if not bucket:
        E5.write_csv(csv_path, [], SUMMARY_FIELDS)
        return
    images = torch.cat([item["images"] for item in bucket], dim=0)
    E1.save_image_grid(images, grid_path, resize=args.preview_size, columns=2)
    E5.write_csv(csv_path, [item["row"] for item in bucket], SUMMARY_FIELDS)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs_c2i/PixelGen_XL_without_CFG.yaml")
    parser.add_argument("--ckpt", default="ckpts/PixelGen_XL_80ep.ckpt")
    parser.add_argument("--output-dir", default="outputs/e5_5_multi_skip_pis")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--classes", default=None, help="Comma-separated class ids. Defaults to 0..num_samples-1.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to seed_start..")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--windows-csv", default=None)
    parser.add_argument("--windows-preset", choices=["smoke", "minimal", "main8", "full", "expanded"], default="minimal")
    parser.add_argument("--total-calls", type=int, default=99, help="Used only with --write-windows-only.")
    parser.add_argument("--write-windows-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--floor-samples", type=int, default=1)
    parser.add_argument("--e5-summary-csv", default="outputs/e5_pis_bank/e5_main8_fullcalls_fp32/pis_summary.csv")
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
    parser.add_argument("--save-image-count", type=int, default=24)
    parser.add_argument("--top-visual-count", type=int, default=32)
    parser.add_argument("--safe-visual-count", type=int, default=16)
    parser.add_argument("--safe-long-min-len", type=int, default=4)
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
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = (PROJECT_ROOT / args.output_dir / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_windows_only:
        total_calls = int(args.total_calls)
        windows = (
            load_windows_csv((PROJECT_ROOT / args.windows_csv).resolve(), total_calls)
            if args.windows_csv
            else generate_windows(args.windows_preset, total_calls)
        )
        write_windows_csv(output_dir / "windows.csv", windows)
        print(f"[E5.5] Wrote {len(windows)} windows to {output_dir / 'windows.csv'}", flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. E5.5 requires a GPU-visible shell.")
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
        raise ValueError("E5.5 multi-skip replay currently expects exact_henu=True.")

    prefixes = [item for item in args.weight_prefixes.split(",") if item != ""]
    if args.weight_prefixes.endswith(","):
        prefixes.append("")
    load_info = E1.load_denoiser_weights(denoiser, ckpt_path, prefixes)
    denoiser.to(device).eval()

    lpips_model, lpips_source = E2.load_lpips_model(args, device)
    dino_model, dino_source = E2.load_dino_model(args, device)

    total_calls = E1.total_denoiser_opportunities(sampler)
    call_timesteps, call_steps, call_kinds = E5.build_call_metadata(sampler)
    if total_calls != call_timesteps.shape[0]:
        raise RuntimeError(f"Expected {total_calls} call metadata entries, got {call_timesteps.shape[0]}")

    windows = (
        load_windows_csv((PROJECT_ROOT / args.windows_csv).resolve(), total_calls)
        if args.windows_csv
        else generate_windows(args.windows_preset, total_calls)
    )
    if not windows:
        raise ValueError("No E5.5 windows selected.")
    selected_windows = shard_windows(windows, args.shard_id, args.num_shards)
    if not selected_windows:
        raise ValueError("This shard received zero windows.")

    write_windows_csv(output_dir / "windows.csv", windows)
    write_windows_csv(output_dir / "selected_windows.csv", selected_windows)

    selected_window_mask = np.zeros((len(windows),), dtype=bool)
    for window in selected_windows:
        selected_window_mask[int(window["window_id"])] = True

    autocast_dtype = torch.bfloat16 if args.autocast_dtype == "bf16" else torch.float16
    num_samples = len(pairs)
    arrays = init_bank_arrays(num_samples, len(windows))

    summary_csv = output_dir / "pis_window_summary.csv"
    rows: List[Dict[str, Any]]
    completed: set[Tuple[int, int]]
    if args.resume:
        rows, completed = read_existing_summary(summary_csv)
        fill_arrays_from_rows(arrays, rows)
    else:
        rows, completed = [], set()

    floor_rows: List[Dict[str, Any]] = []
    top_failures: List[Dict[str, Any]] = []
    safe_long_skips: List[Dict[str, Any]] = []
    preview_images: List[torch.Tensor] = []
    full_reference_images: List[torch.Tensor] = []
    saved_intervention_images = 0

    start_all = time.perf_counter()
    print(
        f"[E5.5] Running multi-skip PIS on {num_samples} samples, "
        f"{len(selected_windows)}/{len(windows)} windows per sample",
        flush=True,
    )

    for sample_idx, (class_id, seed) in enumerate(pairs):
        sample_start = time.perf_counter()
        print(f"[E5.5] Sample {sample_idx}/{num_samples - 1}: class={class_id} seed={seed}", flush=True)

        noise = E1.make_noise_batch([(class_id, seed)], latent_shape, device, args.noise_scale)
        condition, uncondition = conditioner([class_id], {})

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=not args.no_autocast):
            trace = E5.run_full_trace(
                net=denoiser,
                sampler=sampler,
                noise=noise,
                condition=condition,
                uncondition=uncondition,
            )
            full_image = vae.decode(trace["final"].to(device)).detach().cpu().float()

        ref_image_path = ""
        if sample_idx < args.save_image_count:
            ref_image_path = str(
                output_dir
                / "full_reference_images"
                / f"idx{sample_idx:04d}_class{class_id:04d}_seed{seed}.png"
            )
            E1.save_tensor_image(full_image[0], Path(ref_image_path))
        full_reference_images.append(full_image)
        ref_dino_feat = E5.dino_features(full_image, dino_model, device, args)

        if sample_idx < max(0, args.floor_samples):
            floor_start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=not args.no_autocast):
                floor_trace = E5.run_full_trace(
                    net=denoiser,
                    sampler=sampler,
                    noise=noise,
                    condition=condition,
                    uncondition=uncondition,
                )
                floor_image = vae.decode(floor_trace["final"].to(device)).detach().cpu().float()
            floor_metric = E1.compute_batch_metrics(
                reference=full_image,
                candidate=floor_image,
                device=device,
                lpips_model=lpips_model,
            )
            floor_dino = E5.dino_distance_from_ref(ref_dino_feat, floor_image, dino_model, device, args)
            floor_rows.append(
                {
                    "sample_index": sample_idx,
                    "class_id": int(class_id),
                    "seed": int(seed),
                    "floor_lpips": floor_metric["lpips"][0],
                    "floor_dino": floor_dino,
                    "floor_psnr": floor_metric["psnr_db"][0],
                    "floor_ssim": floor_metric["ssim"][0],
                    "elapsed_sec": time.perf_counter() - floor_start,
                }
            )

        for window in selected_windows:
            window_id = int(window["window_id"])
            if (sample_idx, window_id) in completed:
                continue

            meta = window_metadata(window, call_timesteps, call_steps, call_kinds)
            row: Dict[str, Any] = {
                "sample_index": sample_idx,
                "class_id": int(class_id),
                "seed": int(seed),
                "window_id": window_id,
                "stage": window["stage"],
                "start_call": int(window["start_call"]),
                "end_call": int(window["end_call"]),
                "window_len": int(window["window_len"]),
                **meta,
                "intervention_valid": False,
                "invalid_reason": None,
                "pis_lpips": None,
                "pis_dino": None,
                "pis_psnr": None,
                "pis_ssim": None,
                "suffix_denoiser_calls": 0,
                "skipped_calls": 0,
                "elapsed_sec": None,
                "reference_image": ref_image_path,
                "intervention_image": "",
            }

            intervention_start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=not args.no_autocast):
                candidate_latent, denoiser_calls, skipped_calls = run_multi_skip_suffix(
                    net=denoiser,
                    sampler=sampler,
                    trace=trace,
                    condition=condition,
                    uncondition=uncondition,
                    start_call=int(window["start_call"]),
                    window_len=int(window["window_len"]),
                    device=device,
                )
                candidate_image = vae.decode(candidate_latent.to(device)).detach().cpu().float()

            metric = E1.compute_batch_metrics(
                reference=full_image,
                candidate=candidate_image,
                device=device,
                lpips_model=lpips_model,
            )
            dino_distance = E5.dino_distance_from_ref(ref_dino_feat, candidate_image, dino_model, device, args)
            elapsed = time.perf_counter() - intervention_start

            lpips_value = metric["lpips"][0]
            psnr_value = metric["psnr_db"][0]
            ssim_value = metric["ssim"][0]

            arrays["pis_lpips"][sample_idx, window_id] = np.nan if lpips_value is None else float(lpips_value)
            arrays["pis_dino"][sample_idx, window_id] = np.nan if dino_distance is None else float(dino_distance)
            arrays["pis_psnr"][sample_idx, window_id] = float(psnr_value)
            arrays["pis_ssim"][sample_idx, window_id] = float(ssim_value)
            arrays["valid_mask"][sample_idx, window_id] = True
            arrays["suffix_denoiser_calls"][sample_idx, window_id] = int(denoiser_calls)
            arrays["skipped_calls"][sample_idx, window_id] = int(skipped_calls)

            intervention_path = ""
            if saved_intervention_images < args.save_image_count:
                saved_intervention_images += 1
                preview_images.append(candidate_image)
                intervention_path = str(
                    output_dir
                    / "intervention_images"
                    / (
                        f"idx{sample_idx:04d}_wid{window_id:03d}_"
                        f"s{int(window['start_call']):03d}_L{int(window['window_len']):02d}.png"
                    )
                )
                E1.save_tensor_image(candidate_image[0], Path(intervention_path))

            row.update(
                {
                    "intervention_valid": True,
                    "invalid_reason": "",
                    "pis_lpips": None if lpips_value is None else float(lpips_value),
                    "pis_dino": None if dino_distance is None else float(dino_distance),
                    "pis_psnr": float(psnr_value),
                    "pis_ssim": float(ssim_value),
                    "suffix_denoiser_calls": int(denoiser_calls),
                    "skipped_calls": int(skipped_calls),
                    "elapsed_sec": elapsed,
                    "intervention_image": intervention_path,
                }
            )
            rows.append(row)
            completed.add((sample_idx, window_id))

            update_top_failure(
                top_failures,
                row=row,
                full_image=full_image,
                candidate_image=candidate_image,
                limit=args.top_visual_count,
            )
            update_safe_long(
                safe_long_skips,
                row=row,
                full_image=full_image,
                candidate_image=candidate_image,
                limit=args.safe_visual_count,
                min_len=args.safe_long_min_len,
            )

        arrays_to_save = {
            **arrays,
            "selected_window_mask": selected_window_mask,
            "window_start_call": np.asarray([int(row["start_call"]) for row in windows], dtype=np.int16),
            "window_end_call": np.asarray([int(row["end_call"]) for row in windows], dtype=np.int16),
            "window_len": np.asarray([int(row["window_len"]) for row in windows], dtype=np.int16),
            "window_stage": np.asarray([str(row["stage"]) for row in windows]),
            "call_kind": call_kinds,
            "call_step": call_steps,
            "t_values": call_timesteps,
        }
        E5.save_bank(output_dir / "pis_window_bank_partial.npz", arrays_to_save)
        E5.write_csv(summary_csv, rows, SUMMARY_FIELDS)
        E5.write_csv(output_dir / "full_rerun_floor.csv", floor_rows)
        with open(output_dir / "progress.json", "w", encoding="utf-8") as fp:
            json.dump(
                E5.json_ready(
                    {
                        "completed_samples": sample_idx + 1,
                        "total_samples": num_samples,
                        "completed_interventions": int(arrays["valid_mask"].sum()),
                        "target_interventions_this_shard": int(num_samples * len(selected_windows)),
                        "elapsed_sec": time.perf_counter() - start_all,
                        "last_sample_elapsed_sec": time.perf_counter() - sample_start,
                    }
                ),
                fp,
                indent=2,
            )
        torch.cuda.empty_cache()

    arrays_to_save = {
        **arrays,
        "selected_window_mask": selected_window_mask,
        "window_start_call": np.asarray([int(row["start_call"]) for row in windows], dtype=np.int16),
        "window_end_call": np.asarray([int(row["end_call"]) for row in windows], dtype=np.int16),
        "window_len": np.asarray([int(row["window_len"]) for row in windows], dtype=np.int16),
        "window_stage": np.asarray([str(row["stage"]) for row in windows]),
        "call_kind": call_kinds,
        "call_step": call_steps,
        "t_values": call_timesteps,
    }
    E5.save_bank(output_dir / "pis_window_bank.npz", arrays_to_save)
    E5.write_csv(summary_csv, rows, SUMMARY_FIELDS)
    E5.write_csv(output_dir / "full_rerun_floor.csv", floor_rows)

    aggregate_rows = aggregate_window_rows(rows, windows)
    E5.write_csv(output_dir / "pis_window_aggregate.csv", aggregate_rows, AGGREGATE_FIELDS)
    safe_age_rows = stage_safe_age_rows(aggregate_rows)
    E5.write_csv(
        output_dir / "pis_safe_age_by_stage.csv",
        safe_age_rows,
        ["stage", "start_range", "safe_len_mean", "safe_len_q95", "recommended_max_age_for_e6", "reason"],
    )

    e5_summary_path = (PROJECT_ROOT / args.e5_summary_csv).resolve()
    e5_values = load_e5_summary(e5_summary_path)
    synergy_count, l1_count = write_synergy_with_e5(
        output_dir / "pis_synergy_with_e5.csv",
        output_dir / "l1_reproduce_e5_check.csv",
        rows,
        e5_values,
    )

    heatmaps_written = {
        "heatmap_lpips_mean.png": write_heatmap_png(
            output_dir / "heatmap_lpips_mean.png",
            aggregate_rows,
            value_key="mean_lpips",
            title="E5.5 Continuous Skip PIS: mean LPIPS",
        ),
        "heatmap_lpips_q95.png": write_heatmap_png(
            output_dir / "heatmap_lpips_q95.png",
            aggregate_rows,
            value_key="q95_lpips",
            title="E5.5 Continuous Skip PIS: q95 LPIPS",
        ),
        "heatmap_dino_mean.png": write_heatmap_png(
            output_dir / "heatmap_dino_mean.png",
            aggregate_rows,
            value_key="mean_dino",
            title="E5.5 Continuous Skip PIS: mean DINO",
        ),
        "heatmap_dino_q95.png": write_heatmap_png(
            output_dir / "heatmap_dino_q95.png",
            aggregate_rows,
            value_key="q95_dino",
            title="E5.5 Continuous Skip PIS: q95 DINO",
        ),
    }

    if preview_images:
        E1.save_image_grid(
            torch.cat(preview_images, dim=0),
            output_dir / "intervention_preview_grid.png",
            resize=args.preview_size,
            columns=args.preview_columns,
        )
    if full_reference_images and args.save_image_count > 0:
        E1.save_image_grid(
            torch.cat(full_reference_images[: args.save_image_count], dim=0),
            output_dir / "full_reference_preview_grid.png",
            resize=args.preview_size,
            columns=args.preview_columns,
        )
    write_visual_bucket(
        top_failures,
        grid_path=output_dir / "visual_top_failures.png",
        csv_path=output_dir / "visual_top_failures.csv",
        args=args,
    )
    write_visual_bucket(
        safe_long_skips,
        grid_path=output_dir / "visual_safe_long_skips.png",
        csv_path=output_dir / "visual_safe_long_skips.csv",
        args=args,
    )

    elapsed_all = time.perf_counter() - start_all
    floor_summary = {
        "floor_lpips": E1.summarize_metric(row.get("floor_lpips") for row in floor_rows),
        "floor_dino": E1.summarize_metric(row.get("floor_dino") for row in floor_rows),
        "floor_psnr": E1.summarize_metric(row.get("floor_psnr") for row in floor_rows),
        "floor_ssim": E1.summarize_metric(row.get("floor_ssim") for row in floor_rows),
    }
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
            "windows_preset": args.windows_preset,
            "total_windows": len(windows),
            "selected_windows_this_shard": len(selected_windows),
            "shard_id": int(args.shard_id),
            "num_shards": int(args.num_shards),
            "floor_samples": int(args.floor_samples),
            "allow_fused_sdpa": bool(args.allow_fused_sdpa),
            "no_autocast": bool(args.no_autocast),
            "lpips_source": lpips_source,
            "dino_source": dino_source,
            "dino_feature": args.dino_feature,
            "dino_size": args.dino_size,
            "e5_summary_csv": str(e5_summary_path),
            "e5_synergy_rows": synergy_count,
            "l1_reproduce_e5_rows": l1_count,
            "load_info": load_info,
            "args": vars(args),
        },
        "elapsed_sec": elapsed_all,
        "valid_interventions": int(arrays["valid_mask"].sum()),
        "pis_window_bank": str(output_dir / "pis_window_bank.npz"),
        "windows_csv": str(output_dir / "windows.csv"),
        "selected_windows_csv": str(output_dir / "selected_windows.csv"),
        "pis_window_summary_csv": str(summary_csv),
        "pis_window_aggregate_csv": str(output_dir / "pis_window_aggregate.csv"),
        "pis_safe_age_by_stage_csv": str(output_dir / "pis_safe_age_by_stage.csv"),
        "pis_synergy_with_e5_csv": str(output_dir / "pis_synergy_with_e5.csv"),
        "l1_reproduce_e5_check_csv": str(output_dir / "l1_reproduce_e5_check.csv"),
        "full_rerun_floor_csv": str(output_dir / "full_rerun_floor.csv"),
        "heatmaps_written": heatmaps_written,
        "metric_summary": {
            "pis_lpips": metric_summary(arrays["pis_lpips"]),
            "pis_dino": metric_summary(arrays["pis_dino"]),
            "pis_psnr": metric_summary(arrays["pis_psnr"]),
            "pis_ssim": metric_summary(arrays["pis_ssim"]),
        },
        "floor_summary": floor_summary,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(E5.json_ready(summary), fp, indent=2)

    print(f"[E5.5] Summary written to {output_dir / 'summary.json'}", flush=True)
    print(f"[E5.5] Window PIS bank written to {output_dir / 'pis_window_bank.npz'}", flush=True)


if __name__ == "__main__":
    main()
