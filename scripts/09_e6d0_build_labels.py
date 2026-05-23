#!/usr/bin/env python3
"""Build E6-D0 window labels from E5.5 continuous-skip PIS results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LABEL_COLUMNS = [
    "sample_id",
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
    "pis_lpips",
    "pis_dino",
    "pis_psnr",
    "pis_ssim",
    "pis_total_z",
    "pis_total_rank",
]


def resolve_path(path: str | Path) -> Path:
    item = Path(path).expanduser()
    if item.is_absolute():
        return item
    return (PROJECT_ROOT / item).resolve()


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def finite_stats(values: pd.Series) -> Dict[str, Optional[float]]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {"min": float(arr.min()), "mean": float(arr.mean()), "max": float(arr.max())}


def rankdata_average(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata  # type: ignore

        return rankdata(values, method="average").astype(np.float64)
    except Exception:
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(values.shape[0], dtype=np.float64)
        start = 0
        while start < values.shape[0]:
            end = start + 1
            while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
                end += 1
            avg_rank = 0.5 * (start + 1 + end)
            ranks[order[start:end]] = avg_rank
            start = end
        return ranks


def ranknorm(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.full_like(values, 0.5, dtype=np.float64)
    ranks = rankdata_average(values)
    return (ranks - 0.5) / float(values.size)


def zscore(values: np.ndarray) -> np.ndarray:
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return (values - mean) / (std + 1e-8)


def read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e55-dir",
        default="outputs/e5_5_multi_skip_pis/e5_5_main8_windows70_fp32",
        help="Directory containing E5.5 pis_window_summary.csv and windows.csv.",
    )
    parser.add_argument(
        "--out",
        default="outputs/e6_d0_labels/e55_window_labels.csv",
        help="Output label CSV path.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    e55_dir = resolve_path(args.e55_dir)
    out_path = resolve_path(args.out)
    summary_path = out_path.parent / "labels_summary.json"

    summary_csv = e55_dir / "pis_window_summary.csv"
    windows_csv = e55_dir / "windows.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing E5.5 summary: {summary_csv}")
    if not windows_csv.exists():
        raise FileNotFoundError(f"Missing E5.5 windows: {windows_csv}")

    df = pd.read_csv(summary_csv)
    windows = pd.read_csv(windows_csv)
    selected_windows = read_optional_csv(e55_dir / "selected_windows.csv")

    source_summary_json = e55_dir / "summary.json"
    source_summary: Optional[Dict[str, Any]] = None
    if source_summary_json.exists():
        with open(source_summary_json, "r", encoding="utf-8") as fp:
            source_summary = json.load(fp)

    required = {
        "sample_index",
        "class_id",
        "seed",
        "window_id",
        "stage",
        "start_call",
        "end_call",
        "window_len",
        "intervention_valid",
        "pis_lpips",
        "pis_dino",
        "pis_psnr",
        "pis_ssim",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"pis_window_summary.csv missing required columns: {missing}")

    valid_mask = parse_bool_series(df["intervention_valid"])
    valid = df.loc[valid_mask].copy()

    for key in ["pis_lpips", "pis_dino", "pis_psnr", "pis_ssim"]:
        valid[key] = pd.to_numeric(valid[key], errors="coerce")
    finite_core = np.isfinite(valid["pis_lpips"].to_numpy(dtype=np.float64)) & np.isfinite(
        valid["pis_dino"].to_numpy(dtype=np.float64)
    )
    dropped_nonfinite = int((~finite_core).sum())
    if dropped_nonfinite:
        valid = valid.loc[finite_core].copy()

    lpips = valid["pis_lpips"].to_numpy(dtype=np.float64)
    dino = valid["pis_dino"].to_numpy(dtype=np.float64)
    valid["pis_total_z"] = zscore(lpips) + zscore(dino)
    valid["pis_total_rank"] = ranknorm(lpips) + ranknorm(dino)

    if "sample_id" not in valid.columns:
        valid["sample_id"] = valid["sample_index"]

    for col in LABEL_COLUMNS:
        if col not in valid.columns:
            valid[col] = "" if col.endswith("_kind") or col == "stage" else np.nan

    label_df = valid[LABEL_COLUMNS].copy()
    for col in [
        "sample_id",
        "sample_index",
        "class_id",
        "seed",
        "window_id",
        "start_call",
        "end_call",
        "window_len",
        "start_step",
        "end_step",
        "num_predictor",
        "num_corrector",
    ]:
        label_df[col] = pd.to_numeric(label_df[col], errors="raise").astype(int)

    for col in ["pis_total_z", "pis_total_rank"]:
        values = label_df[col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{col} contains NaN/Inf after label construction")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    label_df.to_csv(out_path, index=False)

    expected_by_windows = int(len(windows) * label_df["sample_index"].nunique())
    selected_count = None if selected_windows is None else int(len(selected_windows))
    summary = {
        "input_dir": str(e55_dir),
        "source_files": {
            "pis_window_summary_csv": str(summary_csv),
            "windows_csv": str(windows_csv),
            "selected_windows_csv": str(e55_dir / "selected_windows.csv")
            if selected_windows is not None
            else None,
            "summary_json": str(source_summary_json) if source_summary_json.exists() else None,
        },
        "output_csv": str(out_path),
        "output_summary_json": str(summary_path),
        "total_source_rows": int(len(df)),
        "valid_intervention_rows": int(valid_mask.sum()),
        "dropped_nonfinite_core_metric_rows": dropped_nonfinite,
        "valid_label_rows": int(len(label_df)),
        "sample_count": int(label_df["sample_index"].nunique()),
        "window_count": int(label_df["window_id"].nunique()),
        "windows_csv_rows": int(len(windows)),
        "selected_windows_csv_rows": selected_count,
        "expected_rows_if_dense": expected_by_windows,
        "metrics": {
            "pis_lpips": finite_stats(label_df["pis_lpips"]),
            "pis_dino": finite_stats(label_df["pis_dino"]),
            "pis_psnr": finite_stats(label_df["pis_psnr"]),
            "pis_ssim": finite_stats(label_df["pis_ssim"]),
            "pis_total_z": finite_stats(label_df["pis_total_z"]),
            "pis_total_rank": finite_stats(label_df["pis_total_rank"]),
        },
        "source_summary_meta": source_summary.get("meta") if isinstance(source_summary, dict) else None,
    }
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(json_ready(summary), fp, indent=2, ensure_ascii=False)

    print(f"[E6-D0 labels] wrote {len(label_df)} valid labels to {out_path}", flush=True)
    print(f"[E6-D0 labels] summary written to {summary_path}", flush=True)
    if len(label_df) != 560:
        print(
            f"[E6-D0 labels] note: valid label count is {len(label_df)}, not 560; "
            "using actual valid rows.",
            flush=True,
        )


if __name__ == "__main__":
    main()
