#!/usr/bin/env python3
"""Compare E4 PMA weight candidate reruns against the existing E4 main run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_items(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def rr_label(value: float) -> str:
    return f"rr{value:.2f}".replace(".", "p")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def load_method_summary(run_dir: Path, method_name: str) -> Dict[str, Any]:
    for row in read_csv(run_dir / "method_summary.csv"):
        if row["name"] == method_name:
            out: Dict[str, Any] = dict(row)
            for key, value in list(out.items()):
                converted = safe_float(value)
                if converted is not None:
                    out[key] = converted
            return out
    raise KeyError(f"Missing {method_name} in {run_dir / 'method_summary.csv'}")


def load_per_sample(run_dir: Path, method_name: str) -> Dict[int, Dict[str, float]]:
    rows = read_csv(run_dir / method_name / "per_sample_metrics.csv")
    out: Dict[int, Dict[str, float]] = {}
    for row in rows:
        out[int(row["index"])] = {
            "psnr_db": float(row["psnr_db"]),
            "ssim": float(row["ssim"]),
            "lpips": float(row["lpips"]),
        }
    return out


def bootstrap_mean_ci(values: Sequence[float], seed: int, num_bootstrap: int) -> Dict[str, Optional[float]]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    if finite.size == 1 or num_bootstrap <= 0:
        mean = float(np.mean(finite))
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": int(finite.size)}
    rng = np.random.default_rng(seed)
    n = finite.size
    boot = np.empty((num_bootstrap,), dtype=np.float64)
    for idx in range(num_bootstrap):
        boot[idx] = float(np.mean(finite[rng.integers(0, n, size=n)]))
    return {
        "mean": float(np.mean(finite)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n": int(n),
    }


def paired_row(
    *,
    candidate_name: str,
    candidate_label: str,
    candidate_run: Path,
    candidate_method: str,
    baseline_name: str,
    baseline_label: str,
    baseline_run: Path,
    baseline_method: str,
    target_rr: float,
    seed: int,
    num_bootstrap: int,
) -> Dict[str, Any]:
    cand_summary = load_method_summary(candidate_run, candidate_method)
    base_summary = load_method_summary(baseline_run, baseline_method)
    cand = load_per_sample(candidate_run, candidate_method)
    base = load_per_sample(baseline_run, baseline_method)
    common = sorted(set(cand) & set(base))

    row: Dict[str, Any] = {
        "candidate": candidate_name,
        "candidate_label": candidate_label,
        "candidate_run": str(candidate_run),
        "candidate_method": candidate_method,
        "baseline": baseline_name,
        "baseline_label": baseline_label,
        "baseline_run": str(baseline_run),
        "baseline_method": baseline_method,
        "target_rr": target_rr,
        "candidate_actual_rr": cand_summary.get("actual_rr"),
        "baseline_actual_rr": base_summary.get("actual_rr"),
        "candidate_psnr_mean": cand_summary.get("psnr_mean"),
        "baseline_psnr_mean": base_summary.get("psnr_mean"),
        "candidate_ssim_mean": cand_summary.get("ssim_mean"),
        "baseline_ssim_mean": base_summary.get("ssim_mean"),
        "candidate_lpips_mean": cand_summary.get("lpips_mean"),
        "baseline_lpips_mean": base_summary.get("lpips_mean"),
        "num_common": len(common),
    }
    for metric in ["psnr_db", "ssim", "lpips"]:
        diffs = [cand[idx][metric] - base[idx][metric] for idx in common]
        stats = bootstrap_mean_ci(diffs, seed=seed, num_bootstrap=num_bootstrap)
        row[f"delta_{metric}_mean"] = stats["mean"]
        row[f"delta_{metric}_ci_low"] = stats["ci_low"]
        row[f"delta_{metric}_ci_high"] = stats["ci_high"]
        row[f"delta_{metric}_n"] = stats["n"]
    return row


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-run",
        default="outputs/e4_oracle_schedule_cache/e4_test_rr030_rr040_rr050_fp32",
        help="Existing E4 main run containing sea/no-gate/original stage-aware baselines.",
    )
    parser.add_argument("--candidate-root", default="outputs/e4_oracle_schedule_cache")
    parser.add_argument(
        "--candidate-runs",
        default="e4_candidate_a_rr030_rr040_rr050_fp32,e4_candidate_b_rr030_rr040_rr050_fp32,e4_candidate_c_rr030_rr040_rr050_fp32",
    )
    parser.add_argument("--candidate-names", default="candidate_a,candidate_b,candidate_c")
    parser.add_argument("--candidate-labels", default="A_soft_early_dino,B_soft_early_dino_lpips,C_sea_heavy_low_rr")
    parser.add_argument("--target-rrs", default="0.30,0.40,0.50")
    parser.add_argument("--output-dir", default="outputs/e4_pma_weight_candidates/comparison")
    parser.add_argument("--run-id", default="compare_candidates_vs_main")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--global-seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    main_run = (PROJECT_ROOT / args.main_run).resolve()
    candidate_root = (PROJECT_ROOT / args.candidate_root).resolve()
    candidate_runs = parse_items(args.candidate_runs)
    candidate_names = parse_items(args.candidate_names)
    candidate_labels = parse_items(args.candidate_labels)
    target_rrs = [float(item) for item in parse_items(args.target_rrs)]
    if not (len(candidate_runs) == len(candidate_names) == len(candidate_labels)):
        raise ValueError("--candidate-runs, --candidate-names, and --candidate-labels must have the same length.")

    output_dir = (PROJECT_ROOT / args.output_dir / args.run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for run_id, candidate_name, candidate_label in zip(candidate_runs, candidate_names, candidate_labels):
        candidate_run = candidate_root / run_id
        for target_rr in target_rrs:
            label = rr_label(target_rr)
            candidate_method = f"pma_stageaware_oracle_{label}"
            baselines = [
                ("sea_oracle", "SEA-oracle", f"sea_oracle_{label}"),
                ("pma_nogate_oracle", "PMA-no-gate", f"pma_nogate_oracle_{label}"),
                ("pma_stageaware_original", "original-stage-aware", f"pma_stageaware_oracle_{label}"),
            ]
            for baseline_name, baseline_label, baseline_method in baselines:
                rows.append(
                    paired_row(
                        candidate_name=candidate_name,
                        candidate_label=candidate_label,
                        candidate_run=candidate_run,
                        candidate_method=candidate_method,
                        baseline_name=baseline_name,
                        baseline_label=baseline_label,
                        baseline_run=main_run,
                        baseline_method=baseline_method,
                        target_rr=target_rr,
                        seed=args.global_seed,
                        num_bootstrap=args.bootstrap_samples,
                    )
                )

    fields = [
        "candidate",
        "candidate_label",
        "baseline",
        "baseline_label",
        "target_rr",
        "candidate_actual_rr",
        "baseline_actual_rr",
        "candidate_psnr_mean",
        "baseline_psnr_mean",
        "delta_psnr_db_mean",
        "delta_psnr_db_ci_low",
        "delta_psnr_db_ci_high",
        "candidate_ssim_mean",
        "baseline_ssim_mean",
        "delta_ssim_mean",
        "delta_ssim_ci_low",
        "delta_ssim_ci_high",
        "candidate_lpips_mean",
        "baseline_lpips_mean",
        "delta_lpips_mean",
        "delta_lpips_ci_low",
        "delta_lpips_ci_high",
        "num_common",
        "candidate_run",
        "baseline_run",
        "candidate_method",
        "baseline_method",
    ]
    write_csv(output_dir / "candidate_pairwise_comparison.csv", rows, fields)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump({"rows": rows, "output_csv": str(output_dir / "candidate_pairwise_comparison.csv")}, fp, indent=2)
    print(f"[E4-compare] Wrote {output_dir / 'candidate_pairwise_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
