#!/usr/bin/env python3
"""Run E3 schedule-level oracle analysis from an E2 distance bank.

E3 is an offline analysis stage. It does not rerun PixelGen. It reads the
full-trajectory Raw/SEA/DINO/LPIPS distance bank from E2, builds matched-RR
oracle refresh schedules, and writes schedule files for the later E4 cache
rerun.

Recommended E3 main:

conda run -n pixelgen python scripts/03_e3_schedule_oracle_analysis.py \
  --distance-bank outputs/e2_distance_bank/e2_main_256_fp32/distance_bank.npz \
  --run-id e3_main_256_from_e2_fp32_calib64
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRIC_KEYS = ("raw", "sea", "dino", "lpips")
ORACLE_METHODS = (
    "uniform",
    "raw_oracle",
    "sea_oracle",
    "dino_oracle",
    "lpips_oracle",
    "pma_nogate_oracle",
    "pma_stageaware_oracle",
)
STAGE_NAMES = ("early", "middle", "late")
KIND_NAMES = {0: "predictor", 1: "corrector"}


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_weight_triplet(value: str) -> Tuple[float, float, float]:
    parts = parse_float_list(value)
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated weights, got {value!r}.")
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_stage_weights(value: str) -> Dict[str, Tuple[float, float, float]]:
    weights: Dict[str, Tuple[float, float, float]] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected stage=sea,dino,lpips entry, got {item!r}.")
        stage, triplet = item.split("=", 1)
        stage = stage.strip().lower()
        if stage not in STAGE_NAMES:
            raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGE_NAMES}.")
        weights[stage] = parse_weight_triplet(triplet)
    missing = [stage for stage in STAGE_NAMES if stage not in weights]
    if missing:
        raise ValueError(f"Missing PMA stage weights for: {', '.join(missing)}")
    return weights


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def sanitize_key(value: str) -> str:
    return (
        value.replace("-", "_")
        .replace(".", "p")
        .replace("/", "_")
        .replace(" ", "_")
        .lower()
    )


def rr_label(target_rr: float) -> str:
    return f"rr{target_rr:.2f}".replace(".", "p")


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def maybe_load_e2_metadata(distance_bank_path: Path) -> Dict[str, Any]:
    e2_dir = distance_bank_path.resolve().parent
    summary = load_json_if_exists(e2_dir / "summary.json")
    metadata = load_json_if_exists(e2_dir / "metadata.json")
    return {
        "summary": summary,
        "metadata": metadata,
        "class_seed_pairs": str(e2_dir / "class_seed_pairs.csv")
        if (e2_dir / "class_seed_pairs.csv").exists()
        else None,
    }


def read_completed_bank(npz: np.lib.npyio.NpzFile) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    completed = npz["completed"].astype(bool) if "completed" in npz.files else None
    if completed is None:
        first = npz[METRIC_KEYS[0]]
        completed = np.ones((first.shape[0],), dtype=bool)

    bank: Dict[str, np.ndarray] = {}
    for key in METRIC_KEYS:
        if key not in npz.files:
            raise KeyError(f"Missing required E2 bank array: {key}")
        bank[key] = np.asarray(npz[key][completed], dtype=np.float64)
    return bank, completed


def validate_bank(bank: Mapping[str, np.ndarray]) -> Tuple[int, int, int]:
    shapes = {key: value.shape for key, value in bank.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Metric arrays have mismatched shapes: {shapes}")
    num_samples, num_distances = next(iter(shapes.values()))
    if num_samples <= 0 or num_distances <= 0:
        raise ValueError(f"Invalid bank shape: samples={num_samples}, distances={num_distances}")
    total_calls = num_distances + 1
    for key, value in bank.items():
        if np.isnan(value).any():
            raise ValueError(f"Metric {key} contains NaNs after completed-sample filtering.")
    return int(num_samples), int(num_distances), int(total_calls)


def completed_indices(completed: np.ndarray) -> np.ndarray:
    return np.flatnonzero(completed)


def build_split_indices(num_samples: int, calibration_size: int, shuffle: bool, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(num_samples, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    calibration_size = max(1, min(int(calibration_size), num_samples))
    calibration = np.sort(indices[:calibration_size])
    test = np.sort(indices[calibration_size:])
    return calibration, test


def transition_stage_labels(
    call_steps: np.ndarray,
    num_distances: int,
    early_frac: float,
    late_start_frac: float,
) -> np.ndarray:
    if call_steps.shape[0] != num_distances + 1:
        raise ValueError("call_steps length must equal num_distances + 1.")
    max_step = max(1, int(np.nanmax(call_steps)))
    cur_steps = call_steps[1 : num_distances + 1].astype(np.float64)
    progress = cur_steps / float(max_step)
    labels = np.full((num_distances,), "middle", dtype=object)
    labels[progress < early_frac] = "early"
    labels[progress >= late_start_frac] = "late"
    return labels


def call_stage_labels(
    call_steps: np.ndarray,
    early_frac: float,
    late_start_frac: float,
) -> np.ndarray:
    max_step = max(1, int(np.nanmax(call_steps)))
    progress = call_steps.astype(np.float64) / float(max_step)
    labels = np.full((call_steps.shape[0],), "middle", dtype=object)
    labels[progress < early_frac] = "early"
    labels[progress >= late_start_frac] = "late"
    return labels


def forced_call_mask(total_calls: int, warmup_calls: int, force_final_call: bool) -> np.ndarray:
    mask = np.zeros((total_calls,), dtype=bool)
    warmup_calls = max(0, min(int(warmup_calls), total_calls))
    if warmup_calls > 0:
        mask[:warmup_calls] = True
    if force_final_call and total_calls > 0:
        mask[-1] = True
    if total_calls > 0:
        mask[0] = True
    return mask


def metric_transform(raw: np.ndarray, key: str, args: argparse.Namespace) -> np.ndarray:
    if key == "sea" and args.sea_log1p:
        return np.log1p(np.maximum(raw, 0.0))
    return np.asarray(raw, dtype=np.float64)


def robust_normalize(
    values: np.ndarray,
    calibration_indices: np.ndarray,
    *,
    eps: float,
    clip_percentile: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    calibration_values = values[calibration_indices].reshape(-1)
    calibration_values = calibration_values[np.isfinite(calibration_values)]
    if calibration_values.size == 0:
        raise ValueError("No finite calibration values available for normalization.")

    median = float(np.median(calibration_values))
    normalized = values / max(median, eps)

    clip_value: Optional[float] = None
    if clip_percentile > 0.0:
        clip_value = float(np.percentile(normalized[calibration_indices].reshape(-1), clip_percentile))
        if math.isfinite(clip_value) and clip_value > 0.0:
            normalized = np.clip(normalized, 0.0, clip_value)
        else:
            clip_value = None

    return normalized.astype(np.float64), {
        "median": median,
        "eps": eps,
        "clip_percentile": clip_percentile if clip_value is not None else None,
        "clip_value": clip_value,
    }


def build_scores(
    bank: Mapping[str, np.ndarray],
    calibration_indices: np.ndarray,
    stage_labels: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    transformed: Dict[str, np.ndarray] = {}
    normalized: Dict[str, np.ndarray] = {}
    stats: Dict[str, Any] = {}

    for key in METRIC_KEYS:
        transformed[key] = metric_transform(bank[key], key, args)
        normalized[key], stats[key] = robust_normalize(
            transformed[key],
            calibration_indices,
            eps=args.normalization_eps,
            clip_percentile=args.clip_percentile,
        )
        stats[key]["transform"] = "log1p" if key == "sea" and args.sea_log1p else "identity"

    nogate_weights = parse_weight_triplet(args.pma_nogate_weights)
    stage_weights = parse_stage_weights(args.pma_stage_weights)

    pma_nogate = (
        nogate_weights[0] * normalized["sea"]
        + nogate_weights[1] * normalized["dino"]
        + nogate_weights[2] * normalized["lpips"]
    )

    pma_stageaware = np.zeros_like(pma_nogate)
    for stage in STAGE_NAMES:
        mask = stage_labels == stage
        weights = stage_weights[stage]
        pma_stageaware[:, mask] = (
            weights[0] * normalized["sea"][:, mask]
            + weights[1] * normalized["dino"][:, mask]
            + weights[2] * normalized["lpips"][:, mask]
        )

    scores = {
        "raw_oracle": normalized["raw"],
        "sea_oracle": normalized["sea"],
        "dino_oracle": normalized["dino"],
        "lpips_oracle": normalized["lpips"],
        "pma_nogate_oracle": pma_nogate,
        "pma_stageaware_oracle": pma_stageaware,
    }
    stats["pma_nogate_weights"] = {
        "sea": nogate_weights[0],
        "dino": nogate_weights[1],
        "lpips": nogate_weights[2],
    }
    stats["pma_stage_weights"] = {
        stage: {"sea": weights[0], "dino": weights[1], "lpips": weights[2]}
        for stage, weights in stage_weights.items()
    }
    return scores, stats


def build_uniform_schedule(
    num_samples: int,
    total_calls: int,
    target_rr: float,
    forced_mask: np.ndarray,
) -> np.ndarray:
    target_refreshes = int(round(total_calls * target_rr))
    target_refreshes = max(1, min(total_calls, target_refreshes))
    forced_indices = np.flatnonzero(forced_mask)
    refresh_indices = set(int(idx) for idx in forced_indices)
    remaining = target_refreshes - len(refresh_indices)

    eligible = [idx for idx in range(total_calls) if idx not in refresh_indices]
    if remaining > 0 and eligible:
        if remaining >= len(eligible):
            refresh_indices.update(eligible)
        elif remaining == 1:
            refresh_indices.add(eligible[len(eligible) // 2])
        else:
            positions = np.linspace(0, len(eligible) - 1, remaining)
            for pos in positions:
                refresh_indices.add(eligible[int(round(float(pos)))])

    schedule = np.zeros((num_samples, total_calls), dtype=bool)
    schedule[:, sorted(refresh_indices)] = True
    return schedule


def build_accumulator_schedule(
    score: np.ndarray,
    delta: float,
    forced_mask: np.ndarray,
) -> np.ndarray:
    num_samples, num_distances = score.shape
    total_calls = num_distances + 1
    if forced_mask.shape[0] != total_calls:
        raise ValueError("forced_mask length does not match total calls.")

    schedule = np.zeros((num_samples, total_calls), dtype=bool)
    for sample_idx in range(num_samples):
        acc = 0.0
        for call_idx in range(total_calls):
            if call_idx == 0 or forced_mask[call_idx]:
                schedule[sample_idx, call_idx] = True
                acc = 0.0
                continue

            acc += float(score[sample_idx, call_idx - 1])
            if acc > delta:
                schedule[sample_idx, call_idx] = True
                acc = 0.0
    return schedule


def refresh_ratio(schedule: np.ndarray, indices: Optional[np.ndarray] = None) -> float:
    if indices is not None:
        schedule = schedule[indices]
    if schedule.size == 0:
        return math.nan
    return float(schedule.mean())


def refreshes_per_sample(schedule: np.ndarray, indices: Optional[np.ndarray] = None) -> float:
    if indices is not None:
        schedule = schedule[indices]
    if schedule.shape[0] == 0:
        return math.nan
    return float(schedule.sum(axis=1).mean())


def find_threshold_for_target_rr(
    score: np.ndarray,
    target_rr: float,
    calibration_indices: np.ndarray,
    forced_mask: np.ndarray,
    search_iters: int,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    calibration_score = score[calibration_indices]
    non_forced_distance_mask = ~forced_mask[1:]
    positive = calibration_score[:, non_forced_distance_mask]
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    max_total = float(np.max(np.sum(calibration_score[:, non_forced_distance_mask], axis=1)))
    min_positive = float(np.min(positive)) if positive.size else 0.0

    if max_total <= 0.0:
        schedule = build_accumulator_schedule(score, math.inf, forced_mask)
        return math.inf, schedule, {
            "calibration_rr": refresh_ratio(schedule, calibration_indices),
            "target_rr_reachable": False,
            "reason": "all calibration scores are zero",
        }

    lo = 0.0
    hi = max_total + max(min_positive, 1e-12)
    best_delta = lo
    best_schedule = build_accumulator_schedule(score, lo, forced_mask)
    best_rr = refresh_ratio(best_schedule, calibration_indices)
    best_error = abs(best_rr - target_rr)

    for delta in [hi]:
        schedule = build_accumulator_schedule(score, delta, forced_mask)
        rr = refresh_ratio(schedule, calibration_indices)
        error = abs(rr - target_rr)
        if error < best_error:
            best_delta, best_schedule, best_rr, best_error = delta, schedule, rr, error

    for _ in range(search_iters):
        mid = 0.5 * (lo + hi)
        schedule = build_accumulator_schedule(score, mid, forced_mask)
        rr = refresh_ratio(schedule, calibration_indices)
        error = abs(rr - target_rr)
        if error < best_error:
            best_delta, best_schedule, best_rr, best_error = mid, schedule, rr, error
        if rr > target_rr:
            lo = mid
        else:
            hi = mid

    return best_delta, best_schedule, {
        "calibration_rr": best_rr,
        "target_rr_reachable": True,
        "abs_error": best_error,
        "min_positive_score": min_positive,
        "max_sample_score_sum": max_total,
    }


def threshold_grid(score: np.ndarray, forced_mask: np.ndarray, selected_deltas: Iterable[float], size: int) -> np.ndarray:
    non_forced_distance_mask = ~forced_mask[1:]
    values = score[:, non_forced_distance_mask]
    positive = values[np.isfinite(values) & (values > 0.0)]
    totals = np.sum(values, axis=1)
    candidates: List[float] = [0.0]
    if positive.size:
        min_positive = float(np.min(positive))
        max_total = float(np.max(totals))
        if max_total > 0.0:
            lo = max(min_positive * 0.25, 1e-12)
            if max_total <= lo:
                candidates.append(max_total)
            else:
                candidates.extend(np.geomspace(lo, max_total, max(2, size - 1)).tolist())
    for delta in selected_deltas:
        if math.isfinite(float(delta)):
            candidates.append(float(delta))
    return np.asarray(sorted(set(round(float(item), 12) for item in candidates)), dtype=np.float64)


def split_stats(
    schedule: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
) -> Dict[str, Dict[str, Optional[float]]]:
    splits = {
        "calibration": calibration_indices,
        "test": test_indices,
        "all": None,
    }
    stats: Dict[str, Dict[str, Optional[float]]] = {}
    for split_name, indices in splits.items():
        stats[split_name] = {
            "actual_rr": safe_float(refresh_ratio(schedule, indices)),
            "refreshes_per_sample": safe_float(refreshes_per_sample(schedule, indices)),
            "num_samples": int(schedule.shape[0] if indices is None else len(indices)),
        }
    return stats


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def average_curve_rows(
    bank: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    call_timesteps: np.ndarray,
    call_steps: np.ndarray,
    call_kinds: np.ndarray,
    stage_labels: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    num_distances = next(iter(bank.values())).shape[1]
    for idx in range(num_distances):
        prev_kind = int(call_kinds[idx])
        cur_kind = int(call_kinds[idx + 1])
        row: Dict[str, Any] = {
            "distance_index": idx,
            "stage": str(stage_labels[idx]),
            "prev_call_step": int(call_steps[idx]),
            "prev_call_kind": prev_kind,
            "prev_call_kind_name": KIND_NAMES.get(prev_kind, str(prev_kind)),
            "prev_timestep": float(call_timesteps[idx]),
            "cur_call_step": int(call_steps[idx + 1]),
            "cur_call_kind": cur_kind,
            "cur_call_kind_name": KIND_NAMES.get(cur_kind, str(cur_kind)),
            "cur_timestep": float(call_timesteps[idx + 1]),
        }
        for key in METRIC_KEYS:
            values = bank[key][:, idx]
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
        for key, value in scores.items():
            values = value[:, idx]
            row[f"{key}_score_mean"] = float(np.mean(values))
            row[f"{key}_score_std"] = float(np.std(values))
        rows.append(row)
    return rows


def schedule_summary_row(
    method: str,
    target_rr: float,
    delta: Optional[float],
    schedule: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    stats = split_stats(schedule, calibration_indices, test_indices)
    row: Dict[str, Any] = {
        "method": method,
        "target_rr": target_rr,
        "selected_delta": delta,
        "calibration_rr": stats["calibration"]["actual_rr"],
        "test_rr": stats["test"]["actual_rr"],
        "all_rr": stats["all"]["actual_rr"],
        "calibration_refreshes_per_sample": stats["calibration"]["refreshes_per_sample"],
        "test_refreshes_per_sample": stats["test"]["refreshes_per_sample"],
        "all_refreshes_per_sample": stats["all"]["refreshes_per_sample"],
        "calibration_samples": stats["calibration"]["num_samples"],
        "test_samples": stats["test"]["num_samples"],
        "all_samples": stats["all"]["num_samples"],
    }
    if extra:
        row.update(extra)
    return row


def density_rows(
    *,
    method: str,
    target_rr: float,
    schedule: np.ndarray,
    call_stages: np.ndarray,
    call_kinds: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    splits: Dict[str, Optional[np.ndarray]] = {
        "calibration": calibration_indices,
        "test": test_indices,
        "all": None,
    }
    stage_rows: List[Dict[str, Any]] = []
    stage_kind_rows: List[Dict[str, Any]] = []
    total_calls = schedule.shape[1]
    for split_name, indices in splits.items():
        split_schedule = schedule if indices is None else schedule[indices]
        if split_schedule.shape[0] == 0:
            continue
        total_refreshes = float(split_schedule.sum())
        for stage in STAGE_NAMES:
            stage_mask = call_stages == stage
            refresh_count = float(split_schedule[:, stage_mask].sum())
            opportunities = float(split_schedule.shape[0] * int(stage_mask.sum()))
            stage_rows.append(
                {
                    "method": method,
                    "target_rr": target_rr,
                    "split": split_name,
                    "stage": stage,
                    "refresh_count": refresh_count,
                    "opportunities": opportunities,
                    "density": refresh_count / opportunities if opportunities > 0 else math.nan,
                    "refresh_share": refresh_count / total_refreshes if total_refreshes > 0 else math.nan,
                }
            )

            for kind_value, kind_name in KIND_NAMES.items():
                mask = stage_mask & (call_kinds == kind_value)
                refresh_count = float(split_schedule[:, mask].sum())
                opportunities = float(split_schedule.shape[0] * int(mask.sum()))
                stage_kind_rows.append(
                    {
                        "method": method,
                        "target_rr": target_rr,
                        "split": split_name,
                        "stage": stage,
                        "call_kind": kind_value,
                        "call_kind_name": kind_name,
                        "refresh_count": refresh_count,
                        "opportunities": opportunities,
                        "density": refresh_count / opportunities if opportunities > 0 else math.nan,
                        "refresh_share": refresh_count / total_refreshes if total_refreshes > 0 else math.nan,
                    }
                )
        if total_calls != len(call_stages):
            raise ValueError("call_stages length mismatch.")
    return stage_rows, stage_kind_rows


def write_heatmap_csv(path: Path, schedule: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_index"] + [f"call_{idx:03d}" for idx in range(schedule.shape[1])]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for sample_idx in range(schedule.shape[0]):
            row = {"sample_index": sample_idx}
            row.update({f"call_{idx:03d}": int(schedule[sample_idx, idx]) for idx in range(schedule.shape[1])})
            writer.writerow(row)


def save_schedule_npz(
    path: Path,
    *,
    schedule: np.ndarray,
    method: str,
    target_rr: float,
    selected_delta: Optional[float],
    call_timesteps: np.ndarray,
    call_steps: np.ndarray,
    call_kinds: np.ndarray,
    completed_sample_indices: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    forced_mask: np.ndarray,
    score: Optional[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, Any] = {
        "schedule": schedule.astype(np.bool_),
        "method": np.asarray(method),
        "target_rr": np.asarray(float(target_rr), dtype=np.float32),
        "selected_delta": np.asarray(math.nan if selected_delta is None else float(selected_delta), dtype=np.float64),
        "actual_rr": np.asarray(refresh_ratio(schedule), dtype=np.float32),
        "call_timesteps": call_timesteps.astype(np.float32),
        "call_steps": call_steps.astype(np.int16),
        "call_kinds": call_kinds.astype(np.int8),
        "completed_sample_indices": completed_sample_indices.astype(np.int64),
        "calibration_indices": calibration_indices.astype(np.int64),
        "test_indices": test_indices.astype(np.int64),
        "forced_mask": forced_mask.astype(np.bool_),
    }
    if score is not None:
        arrays["score"] = score.astype(np.float32)
    np.savez(path, **arrays)


def _svg_palette() -> List[str]:
    return ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563"]


def _finite_min_max(values: Sequence[float]) -> Tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    low, high = min(finite), max(finite)
    if low == high:
        pad = abs(low) * 0.05 + 1.0
        return low - pad, high + pad
    return low, high


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return 0.5 * (dst_min + dst_max)
    alpha = (value - src_min) / (src_max - src_min)
    return dst_min + alpha * (dst_max - dst_min)


def write_svg_line_plot(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    x_values: Sequence[float],
    series: Mapping[str, Sequence[float]],
    log_x: bool = False,
    width: int = 960,
    height: int = 520,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin_left, margin_right, margin_top, margin_bottom = 78, 220, 46, 70
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    def x_transform(value: float) -> float:
        if log_x:
            return math.log10(max(value, 1e-12))
        return value

    x_transformed = [x_transform(float(value)) for value in x_values]
    all_y = [float(value) for values in series.values() for value in values]
    x_min, x_max = _finite_min_max(x_transformed)
    y_min, y_max = _finite_min_max(all_y)
    y_min = min(0.0, y_min)

    palette = _svg_palette()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(6):
        frac = tick / 5
        y = plot_bottom - frac * (plot_bottom - plot_top)
        value = y_min + frac * (y_max - y_min)
        parts.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{plot_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#374151">{value:.3g}</text>'
        )
    for tick in range(6):
        frac = tick / 5
        x = plot_left + frac * (plot_right - plot_left)
        value = x_min + frac * (x_max - x_min)
        label = f"{10 ** value:.3g}" if log_x else f"{value:.3g}"
        parts.append(f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom + 5}" stroke="#111827"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 22}" text-anchor="middle" font-family="Arial" font-size="11" fill="#374151">{label}</text>'
        )

    for idx, (name, values) in enumerate(series.items()):
        color = palette[idx % len(palette)]
        points = []
        for x_raw, y_raw in zip(x_transformed, values):
            if not math.isfinite(float(y_raw)):
                continue
            x = _scale(float(x_raw), x_min, x_max, plot_left, plot_right)
            y = _scale(float(y_raw), y_min, y_max, plot_bottom, plot_top)
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.8" points="{" ".join(points)}"/>'
            )
        legend_y = plot_top + 20 + idx * 20
        parts.append(f'<rect x="{plot_right + 28}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{plot_right + 46}" y="{legend_y}" font-family="Arial" font-size="12" fill="#111827">{html.escape(name)}</text>'
        )

    parts.append(
        f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(xlabel)}</text>'
    )
    parts.append(
        f'<text transform="translate(20 {(plot_top + plot_bottom) / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(ylabel)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_svg_xy_series_plot(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Mapping[str, Tuple[Sequence[float], Sequence[float]]],
    log_x: bool = False,
    width: int = 960,
    height: int = 520,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin_left, margin_right, margin_top, margin_bottom = 78, 220, 46, 70
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    def x_transform(value: float) -> float:
        if log_x:
            return math.log10(max(value, 1e-12))
        return value

    all_x = [x_transform(float(x)) for x_values, _ in series.values() for x in x_values]
    all_y = [float(y) for _, y_values in series.values() for y in y_values]
    x_min, x_max = _finite_min_max(all_x)
    y_min, y_max = _finite_min_max(all_y)
    y_min = min(0.0, y_min)

    palette = _svg_palette()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(6):
        frac = tick / 5
        y = plot_bottom - frac * (plot_bottom - plot_top)
        value = y_min + frac * (y_max - y_min)
        parts.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{plot_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#374151">{value:.3g}</text>'
        )
    for tick in range(6):
        frac = tick / 5
        x = plot_left + frac * (plot_right - plot_left)
        value = x_min + frac * (x_max - x_min)
        label = f"{10 ** value:.3g}" if log_x else f"{value:.3g}"
        parts.append(f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom + 5}" stroke="#111827"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 22}" text-anchor="middle" font-family="Arial" font-size="11" fill="#374151">{label}</text>'
        )

    for idx, (name, (x_values, y_values)) in enumerate(series.items()):
        color = palette[idx % len(palette)]
        pairs = sorted(zip(x_values, y_values), key=lambda item: float(item[0]))
        points = []
        for x_raw, y_raw in pairs:
            if not math.isfinite(float(x_raw)) or not math.isfinite(float(y_raw)):
                continue
            x = _scale(x_transform(float(x_raw)), x_min, x_max, plot_left, plot_right)
            y = _scale(float(y_raw), y_min, y_max, plot_bottom, plot_top)
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.8" points="{" ".join(points)}"/>'
            )
        legend_y = plot_top + 20 + idx * 20
        parts.append(f'<rect x="{plot_right + 28}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{plot_right + 46}" y="{legend_y}" font-family="Arial" font-size="12" fill="#111827">{html.escape(name)}</text>'
        )

    parts.append(
        f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(xlabel)}</text>'
    )
    parts.append(
        f'<text transform="translate(20 {(plot_top + plot_bottom) / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(ylabel)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_svg_heatmap(path: Path, *, title: str, schedule: np.ndarray, width: int = 940, height: int = 520) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin_left, margin_right, margin_top, margin_bottom = 74, 28, 42, 54
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    num_samples, total_calls = schedule.shape
    cell_w = (plot_right - plot_left) / total_calls
    cell_h = (plot_bottom - plot_top) / num_samples
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_right - plot_left}" height="{plot_bottom - plot_top}" fill="#f3f4f6" stroke="#111827" stroke-width="1"/>',
    ]
    ys, xs = np.nonzero(schedule)
    for sample_idx, call_idx in zip(ys, xs):
        x = plot_left + call_idx * cell_w
        y = plot_top + sample_idx * cell_h
        parts.append(
            f'<rect x="{x:.3f}" y="{y:.3f}" width="{max(cell_w, 0.6):.3f}" height="{max(cell_h, 0.6):.3f}" fill="#7c3aed"/>'
        )
    for tick in range(6):
        frac = tick / 5
        x = plot_left + frac * (plot_right - plot_left)
        call_label = int(round(frac * (total_calls - 1)))
        parts.append(f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom + 5}" stroke="#111827"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 22}" text-anchor="middle" font-family="Arial" font-size="11">{call_label}</text>'
        )
    for tick in range(5):
        frac = tick / 4
        y = plot_top + frac * (plot_bottom - plot_top)
        sample_label = int(round(frac * (num_samples - 1)))
        parts.append(f'<line x1="{plot_left - 5}" y1="{y:.1f}" x2="{plot_left}" y2="{y:.1f}" stroke="#111827"/>')
        parts.append(
            f'<text x="{plot_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{sample_label}</text>'
        )
    parts.append(
        f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13">call index</text>'
    )
    parts.append(
        f'<text transform="translate(20 {(plot_top + plot_bottom) / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">sample index</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_svg_stage_density(
    path: Path,
    *,
    title: str,
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    target_rr: float,
    width: int = 960,
    height: int = 520,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin_left, margin_right, margin_top, margin_bottom = 72, 36, 46, 100
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    values: Dict[Tuple[str, str], float] = {}
    for row in rows:
        if row["split"] == "all" and float(row["target_rr"]) == float(target_rr):
            values[(str(row["method"]), str(row["stage"]))] = float(row["density"])
    max_y = max([0.01] + list(values.values()))
    palette = {"early": "#2563eb", "middle": "#16a34a", "late": "#dc2626"}
    group_w = (plot_right - plot_left) / max(1, len(methods))
    bar_w = group_w * 0.22
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#111827"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#111827"/>',
    ]
    for tick in range(6):
        frac = tick / 5
        y = plot_bottom - frac * (plot_bottom - plot_top)
        value = frac * max_y
        parts.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{plot_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>'
        )
    for method_idx, method in enumerate(methods):
        center = plot_left + group_w * (method_idx + 0.5)
        for stage_idx, stage in enumerate(STAGE_NAMES):
            value = values.get((method, stage), 0.0)
            x = center + (stage_idx - 1) * bar_w - 0.5 * bar_w
            y = _scale(value, 0.0, max_y, plot_bottom, plot_top)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{plot_bottom - y:.1f}" fill="{palette[stage]}"/>'
            )
        parts.append(
            f'<text x="{center:.1f}" y="{plot_bottom + 18}" text-anchor="end" transform="rotate(-25 {center:.1f} {plot_bottom + 18})" font-family="Arial" font-size="11">{html.escape(method)}</text>'
        )
    for idx, stage in enumerate(STAGE_NAMES):
        x = plot_right - 170 + idx * 70
        parts.append(f'<rect x="{x}" y="{plot_top + 6}" width="12" height="12" fill="{palette[stage]}"/>')
        parts.append(f'<text x="{x + 18}" y="{plot_top + 17}" font-family="Arial" font-size="12">{stage}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_svg_fallback_plots(
    *,
    output_dir: Path,
    average_rows: Sequence[Mapping[str, Any]],
    schedules: Mapping[Tuple[str, float], np.ndarray],
    stage_rows: Sequence[Mapping[str, Any]],
    threshold_rows: Sequence[Mapping[str, Any]],
) -> str:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    x_values = [float(row["distance_index"]) for row in average_rows]
    average_series = {
        key: [float(row[f"{key}_score_mean"]) for row in average_rows]
        for key in ["raw_oracle", "sea_oracle", "dino_oracle", "lpips_oracle", "pma_stageaware_oracle"]
    }
    write_svg_line_plot(
        plots_dir / "average_metric_curves.svg",
        title="E3 average metric score curves",
        xlabel="distance index",
        ylabel="mean normalized score",
        x_values=x_values,
        series=average_series,
    )

    threshold_series: Dict[str, Tuple[List[float], List[float]]] = {}
    for method in [item for item in ORACLE_METHODS if item != "uniform"]:
        rows = [row for row in threshold_rows if row["method"] == method and row["split"] == "calibration"]
        rows = sorted(rows, key=lambda row: float(row["threshold"]))
        if not rows:
            continue
        xs = [float(row["threshold"]) for row in rows]
        ys = [float(row["actual_rr"]) for row in rows]
        threshold_series[method] = (xs, ys)
    if threshold_series:
        write_svg_xy_series_plot(
            plots_dir / "threshold_vs_rr.svg",
            title="E3 threshold vs achieved RR",
            xlabel="threshold delta",
            ylabel="calibration RR",
            series=threshold_series,
            log_x=True,
        )

    for (method, target_rr), schedule in schedules.items():
        if method not in {"uniform", "sea_oracle", "pma_nogate_oracle", "pma_stageaware_oracle"}:
            continue
        write_svg_heatmap(
            plots_dir / f"refresh_heatmap_{sanitize_key(method)}_{rr_label(target_rr)}.svg",
            title=f"{method} {rr_label(target_rr)} refresh heatmap",
            schedule=schedule,
        )

    density_methods = ["uniform", "raw_oracle", "sea_oracle", "pma_stageaware_oracle"]
    for target_rr in sorted({float(row["target_rr"]) for row in stage_rows}):
        write_svg_stage_density(
            plots_dir / f"stage_refresh_density_{rr_label(target_rr)}.svg",
            title=f"E3 stage refresh density {rr_label(target_rr)}",
            rows=stage_rows,
            methods=density_methods,
            target_rr=target_rr,
        )
    return "svg fallback written"


def maybe_write_plots(
    *,
    output_dir: Path,
    average_rows: Sequence[Mapping[str, Any]],
    schedules: Mapping[Tuple[str, float], np.ndarray],
    stage_rows: Sequence[Mapping[str, Any]],
    threshold_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Optional[str]:
    if args.no_plots:
        return "disabled"
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        fallback = write_svg_fallback_plots(
            output_dir=output_dir,
            average_rows=average_rows,
            schedules=schedules,
            stage_rows=stage_rows,
            threshold_rows=threshold_rows,
        )
        return f"matplotlib unavailable: {exc}; {fallback}"

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    x = np.asarray([int(row["distance_index"]) for row in average_rows])
    plt.figure(figsize=(11, 5))
    for key in ["raw_oracle", "sea_oracle", "dino_oracle", "lpips_oracle", "pma_stageaware_oracle"]:
        y = np.asarray([float(row[f"{key}_score_mean"]) for row in average_rows])
        plt.plot(x, y, label=key)
    plt.xlabel("distance index")
    plt.ylabel("mean normalized score")
    plt.title("E3 average metric score curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "average_metric_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for method in [item for item in ORACLE_METHODS if item != "uniform"]:
        rows = [row for row in threshold_rows if row["method"] == method and row["split"] == "calibration"]
        if not rows:
            continue
        thresholds = np.asarray([float(row["threshold"]) for row in rows])
        rr = np.asarray([float(row["actual_rr"]) for row in rows])
        order = np.argsort(thresholds)
        plt.plot(thresholds[order], rr[order], label=method)
    plt.xscale("symlog", linthresh=1e-3)
    plt.xlabel("threshold delta")
    plt.ylabel("calibration RR")
    plt.title("E3 threshold vs achieved RR")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plots_dir / "threshold_vs_rr.png", dpi=180)
    plt.close()

    for (method, target_rr), schedule in schedules.items():
        if method not in {"uniform", "sea_oracle", "pma_nogate_oracle", "pma_stageaware_oracle"}:
            continue
        plt.figure(figsize=(11, 5))
        plt.imshow(schedule.astype(np.float32), aspect="auto", interpolation="nearest", cmap="magma")
        plt.xlabel("call index")
        plt.ylabel("sample index")
        plt.title(f"{method} {rr_label(target_rr)} refresh heatmap")
        plt.colorbar(label="refresh")
        plt.tight_layout()
        plt.savefig(plots_dir / f"refresh_heatmap_{sanitize_key(method)}_{rr_label(target_rr)}.png", dpi=180)
        plt.close()

    methods_for_density = ["uniform", "raw_oracle", "sea_oracle", "pma_stageaware_oracle"]
    for target_rr in sorted({float(row["target_rr"]) for row in stage_rows}):
        rows = [
            row
            for row in stage_rows
            if row["split"] == "all" and float(row["target_rr"]) == target_rr and row["method"] in methods_for_density
        ]
        if not rows:
            continue
        x_positions = np.arange(len(methods_for_density), dtype=np.float64)
        width = 0.25
        plt.figure(figsize=(10, 5))
        for offset, stage in enumerate(STAGE_NAMES):
            values = []
            for method in methods_for_density:
                match = [row for row in rows if row["method"] == method and row["stage"] == stage]
                values.append(float(match[0]["density"]) if match else 0.0)
            plt.bar(x_positions + (offset - 1) * width, values, width=width, label=stage)
        plt.xticks(x_positions, methods_for_density, rotation=15, ha="right")
        plt.ylabel("refresh density")
        plt.title(f"E3 stage refresh density {rr_label(target_rr)}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"stage_refresh_density_{rr_label(target_rr)}.png", dpi=180)
        plt.close()

    return "written"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distance-bank",
        default="outputs/e2_distance_bank/e2_main_256_fp32/distance_bank.npz",
        help="Path to E2 distance_bank.npz.",
    )
    parser.add_argument("--output-dir", default="outputs/e3_schedule_oracle")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--target-rrs", default="0.30,0.40,0.50")
    parser.add_argument("--calibration-size", type=int, default=64)
    parser.add_argument("--shuffle-calibration", action="store_true")
    parser.add_argument("--global-seed", type=int, default=2026)
    parser.add_argument("--warmup-calls", type=int, default=5)
    parser.add_argument("--no-force-final-call", action="store_true")
    parser.add_argument("--early-frac", type=float, default=0.30)
    parser.add_argument("--late-start-frac", type=float, default=0.70)
    parser.add_argument("--sea-log1p", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalization-eps", type=float, default=1e-8)
    parser.add_argument("--clip-percentile", type=float, default=99.0)
    parser.add_argument("--threshold-search-iters", type=int, default=80)
    parser.add_argument("--threshold-grid-size", type=int, default=80)
    parser.add_argument("--pma-nogate-weights", default="0.4,0.3,0.3")
    parser.add_argument(
        "--pma-stage-weights",
        default="early=1.0,0.0,0.0;middle=0.5,0.5,0.0;late=0.25,0.35,0.40",
        help=(
            "Stage weights as early=sea,dino,lpips;middle=...;late=... "
            "The default follows pixeldiffusion_phase1_plan_revised.md Section 4.6."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)

    distance_bank_path = (PROJECT_ROOT / args.distance_bank).resolve()
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = (PROJECT_ROOT / args.output_dir / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_rrs = parse_float_list(args.target_rrs)
    if not target_rrs:
        raise ValueError("At least one target RR is required.")
    if any(rr <= 0.0 or rr > 1.0 for rr in target_rrs):
        raise ValueError("--target-rrs must be in (0, 1].")
    if not (0.0 < args.early_frac < args.late_start_frac < 1.0):
        raise ValueError("Expected 0 < early_frac < late_start_frac < 1.")

    started_at = time.perf_counter()
    print(f"[E3] Loading distance bank: {distance_bank_path}", flush=True)
    with np.load(distance_bank_path) as npz:
        bank, completed_mask = read_completed_bank(npz)
        call_timesteps = np.asarray(npz["call_timesteps"], dtype=np.float64)
        call_steps = np.asarray(npz["call_steps"], dtype=np.int16)
        call_kinds = np.asarray(npz["call_kinds"], dtype=np.int8)

    num_samples, num_distances, total_calls = validate_bank(bank)
    if call_timesteps.shape[0] != total_calls or call_steps.shape[0] != total_calls or call_kinds.shape[0] != total_calls:
        raise ValueError("Call metadata length mismatch.")

    calibration_indices, test_indices = build_split_indices(
        num_samples,
        calibration_size=args.calibration_size,
        shuffle=args.shuffle_calibration,
        seed=args.global_seed,
    )
    completed_sample_indices = completed_indices(completed_mask)
    transition_stages = transition_stage_labels(
        call_steps,
        num_distances,
        early_frac=args.early_frac,
        late_start_frac=args.late_start_frac,
    )
    call_stages = call_stage_labels(
        call_steps,
        early_frac=args.early_frac,
        late_start_frac=args.late_start_frac,
    )
    forced_mask = forced_call_mask(
        total_calls,
        warmup_calls=args.warmup_calls,
        force_final_call=not args.no_force_final_call,
    )

    print(
        f"[E3] samples={num_samples}, calls/sample={total_calls}, distances/sample={num_distances}, "
        f"calibration={len(calibration_indices)}, test={len(test_indices)}",
        flush=True,
    )

    scores, normalization = build_scores(bank, calibration_indices, transition_stages, args)
    average_rows = average_curve_rows(bank, scores, call_timesteps, call_steps, call_kinds, transition_stages)
    average_fields = list(average_rows[0].keys()) if average_rows else []
    write_csv(output_dir / "average_metric_curves.csv", average_rows, average_fields)

    schedule_dir = output_dir / "matched_schedules"
    heatmap_dir = output_dir / "refresh_heatmaps"
    schedule_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    stage_rows: List[Dict[str, Any]] = []
    stage_kind_rows: List[Dict[str, Any]] = []
    schedules_for_plots: Dict[Tuple[str, float], np.ndarray] = {}
    selected_deltas: Dict[str, Dict[str, float]] = {}

    for target_rr in target_rrs:
        uniform_schedule = build_uniform_schedule(num_samples, total_calls, target_rr, forced_mask)
        schedules_for_plots[("uniform", target_rr)] = uniform_schedule
        schedule_rows.append(
            schedule_summary_row("uniform", target_rr, None, uniform_schedule, calibration_indices, test_indices)
        )
        stage_density, stage_kind_density = density_rows(
            method="uniform",
            target_rr=target_rr,
            schedule=uniform_schedule,
            call_stages=call_stages,
            call_kinds=call_kinds,
            calibration_indices=calibration_indices,
            test_indices=test_indices,
        )
        stage_rows.extend(stage_density)
        stage_kind_rows.extend(stage_kind_density)
        write_heatmap_csv(heatmap_dir / f"refresh_heatmap_uniform_{rr_label(target_rr)}.csv", uniform_schedule)
        save_schedule_npz(
            schedule_dir / f"uniform_{rr_label(target_rr)}.npz",
            schedule=uniform_schedule,
            method="uniform",
            target_rr=target_rr,
            selected_delta=None,
            call_timesteps=call_timesteps,
            call_steps=call_steps,
            call_kinds=call_kinds,
            completed_sample_indices=completed_sample_indices,
            calibration_indices=calibration_indices,
            test_indices=test_indices,
            forced_mask=forced_mask,
            score=None,
        )

    for method, score in scores.items():
        selected_deltas[method] = {}
        per_method_selected: List[float] = []
        method_schedules: Dict[float, Tuple[np.ndarray, float, Dict[str, Any]]] = {}

        print(f"[E3] Searching thresholds for {method}", flush=True)
        for target_rr in target_rrs:
            delta, schedule, info = find_threshold_for_target_rr(
                score,
                target_rr,
                calibration_indices,
                forced_mask,
                search_iters=args.threshold_search_iters,
            )
            selected_deltas[method][rr_label(target_rr)] = float(delta)
            per_method_selected.append(float(delta))
            method_schedules[target_rr] = (schedule, float(delta), info)

        for threshold in threshold_grid(score, forced_mask, per_method_selected, size=args.threshold_grid_size):
            schedule = build_accumulator_schedule(score, float(threshold), forced_mask)
            stats = split_stats(schedule, calibration_indices, test_indices)
            for split_name, split_stat in stats.items():
                threshold_rows.append(
                    {
                        "method": method,
                        "threshold": float(threshold),
                        "split": split_name,
                        "actual_rr": split_stat["actual_rr"],
                        "refreshes_per_sample": split_stat["refreshes_per_sample"],
                        "num_samples": split_stat["num_samples"],
                    }
                )

        for target_rr, (schedule, delta, info) in method_schedules.items():
            schedules_for_plots[(method, target_rr)] = schedule
            schedule_rows.append(
                schedule_summary_row(
                    method,
                    target_rr,
                    delta,
                    schedule,
                    calibration_indices,
                    test_indices,
                    extra={
                        "target_abs_error": info.get("abs_error"),
                        "target_rr_reachable": info.get("target_rr_reachable"),
                        "min_positive_score": info.get("min_positive_score"),
                        "max_sample_score_sum": info.get("max_sample_score_sum"),
                    },
                )
            )
            stage_density, stage_kind_density = density_rows(
                method=method,
                target_rr=target_rr,
                schedule=schedule,
                call_stages=call_stages,
                call_kinds=call_kinds,
                calibration_indices=calibration_indices,
                test_indices=test_indices,
            )
            stage_rows.extend(stage_density)
            stage_kind_rows.extend(stage_kind_density)
            write_heatmap_csv(heatmap_dir / f"refresh_heatmap_{sanitize_key(method)}_{rr_label(target_rr)}.csv", schedule)
            save_schedule_npz(
                schedule_dir / f"{sanitize_key(method)}_{rr_label(target_rr)}.npz",
                schedule=schedule,
                method=method,
                target_rr=target_rr,
                selected_delta=delta,
                call_timesteps=call_timesteps,
                call_steps=call_steps,
                call_kinds=call_kinds,
                completed_sample_indices=completed_sample_indices,
                calibration_indices=calibration_indices,
                test_indices=test_indices,
                forced_mask=forced_mask,
                score=score,
            )

    schedule_fields = [
        "method",
        "target_rr",
        "selected_delta",
        "calibration_rr",
        "test_rr",
        "all_rr",
        "calibration_refreshes_per_sample",
        "test_refreshes_per_sample",
        "all_refreshes_per_sample",
        "calibration_samples",
        "test_samples",
        "all_samples",
        "target_abs_error",
        "target_rr_reachable",
        "min_positive_score",
        "max_sample_score_sum",
    ]
    write_csv(output_dir / "schedule_summary.csv", schedule_rows, schedule_fields)

    threshold_fields = ["method", "threshold", "split", "actual_rr", "refreshes_per_sample", "num_samples"]
    write_csv(output_dir / "threshold_vs_rr.csv", threshold_rows, threshold_fields)

    stage_fields = [
        "method",
        "target_rr",
        "split",
        "stage",
        "refresh_count",
        "opportunities",
        "density",
        "refresh_share",
    ]
    write_csv(output_dir / "stage_refresh_density.csv", stage_rows, stage_fields)

    stage_kind_fields = [
        "method",
        "target_rr",
        "split",
        "stage",
        "call_kind",
        "call_kind_name",
        "refresh_count",
        "opportunities",
        "density",
        "refresh_share",
    ]
    write_csv(output_dir / "stage_kind_refresh_density.csv", stage_kind_rows, stage_kind_fields)

    plot_status = maybe_write_plots(
        output_dir=output_dir,
        average_rows=average_rows,
        schedules=schedules_for_plots,
        stage_rows=stage_rows,
        threshold_rows=threshold_rows,
        args=args,
    )

    e2_meta = maybe_load_e2_metadata(distance_bank_path)
    metadata = {
        "experiment": "E3 schedule-level oracle analysis",
        "distance_bank": str(distance_bank_path),
        "output_dir": str(output_dir),
        "e2": e2_meta,
        "num_samples": num_samples,
        "num_distances": num_distances,
        "denoiser_opportunities_per_sample": total_calls,
        "target_rrs": target_rrs,
        "calibration_indices": calibration_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "completed_sample_indices": completed_sample_indices.tolist(),
        "forced_call_indices": np.flatnonzero(forced_mask).astype(int).tolist(),
        "stage_definition": {
            "mode": "call_step_fraction",
            "early_frac": args.early_frac,
            "late_start_frac": args.late_start_frac,
            "transition_stage_counts": {
                stage: int(np.sum(transition_stages == stage)) for stage in STAGE_NAMES
            },
            "call_stage_counts": {
                stage: int(np.sum(call_stages == stage)) for stage in STAGE_NAMES
            },
        },
        "normalization": normalization,
        "selected_deltas": selected_deltas,
        "plots": plot_status,
        "args": vars(args),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    summary = {
        "metadata": str(output_dir / "metadata.json"),
        "schedule_summary": str(output_dir / "schedule_summary.csv"),
        "threshold_vs_rr": str(output_dir / "threshold_vs_rr.csv"),
        "average_metric_curves": str(output_dir / "average_metric_curves.csv"),
        "stage_refresh_density": str(output_dir / "stage_refresh_density.csv"),
        "stage_kind_refresh_density": str(output_dir / "stage_kind_refresh_density.csv"),
        "matched_schedules": str(schedule_dir),
        "refresh_heatmaps": str(heatmap_dir),
        "elapsed_sec": time.perf_counter() - started_at,
        "schedule_rows": len(schedule_rows),
        "threshold_rows": len(threshold_rows),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[E3] Schedule summary written to {output_dir / 'schedule_summary.csv'}", flush=True)
    print(f"[E3] Matched schedules written to {schedule_dir}", flush=True)
    print(f"[E3] Summary written to {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
