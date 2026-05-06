#!/usr/bin/env python3
"""Prepare PMA stage-aware weight candidate schedules for E4 reruns.

This is a lightweight CPU-only helper. It reuses the E2 distance bank and the
E3 threshold-search convention, then writes candidate-specific
``matched_schedules`` directories that can be consumed by
``04_e4_oracle_schedule_cache_rerun.py``.
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
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_e3_helpers() -> Any:
    path = PROJECT_ROOT / "scripts" / "03_e3_schedule_oracle_analysis.py"
    spec = importlib.util.spec_from_file_location("pixelgen_e3_schedule_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import E3 helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E3 = load_e3_helpers()


CANDIDATE_DEFAULTS = {
    "candidate_a": {
        "label": "A_soft_early_dino",
        "weights": "early=0.75,0.25,0.00;middle=0.45,0.45,0.10;late=0.25,0.35,0.40",
    },
    "candidate_b": {
        "label": "B_soft_early_dino_lpips",
        "weights": "early=0.70,0.20,0.10;middle=0.45,0.35,0.20;late=0.30,0.30,0.40",
    },
    "candidate_c": {
        "label": "C_sea_heavy_low_rr",
        "weights": "early=0.85,0.15,0.00;middle=0.55,0.35,0.10;late=0.35,0.30,0.35",
    },
}


def parse_candidate_keys(value: str) -> List[str]:
    keys = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(keys) - set(CANDIDATE_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}. Allowed: {sorted(CANDIDATE_DEFAULTS)}")
    return keys


def json_ready(value: Any) -> Any:
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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        seen: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fields = seen
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-bank", default="outputs/e2_distance_bank/e2_main_256_fp32/distance_bank.npz")
    parser.add_argument("--output-dir", default="outputs/e4_pma_weight_candidates")
    parser.add_argument("--run-id", default="e4_pma_weight_candidates_from_e2_fp32_calib64")
    parser.add_argument("--candidates", default="candidate_a,candidate_b,candidate_c")
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
    parser.add_argument("--pma-nogate-weights", default="0.4,0.3,0.3")
    parser.add_argument("--candidate-a-weights", default=CANDIDATE_DEFAULTS["candidate_a"]["weights"])
    parser.add_argument("--candidate-b-weights", default=CANDIDATE_DEFAULTS["candidate_b"]["weights"])
    parser.add_argument("--candidate-c-weights", default=CANDIDATE_DEFAULTS["candidate_c"]["weights"])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    target_rrs = E3.parse_float_list(args.target_rrs)
    candidate_keys = parse_candidate_keys(args.candidates)
    candidate_weights = {
        "candidate_a": args.candidate_a_weights,
        "candidate_b": args.candidate_b_weights,
        "candidate_c": args.candidate_c_weights,
    }

    distance_bank_path = (PROJECT_ROOT / args.distance_bank).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir / args.run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    with np.load(distance_bank_path) as npz:
        bank, completed_mask = E3.read_completed_bank(npz)
        call_timesteps = np.asarray(npz["call_timesteps"], dtype=np.float32)
        call_steps = np.asarray(npz["call_steps"], dtype=np.int16)
        call_kinds = np.asarray(npz["call_kinds"], dtype=np.int8)

    num_samples, num_distances, total_calls = E3.validate_bank(bank)
    if call_timesteps.shape[0] != total_calls or call_steps.shape[0] != total_calls or call_kinds.shape[0] != total_calls:
        raise ValueError("call_timesteps/call_steps/call_kinds length does not match distance bank.")

    calibration_indices, test_indices = E3.build_split_indices(
        num_samples,
        calibration_size=args.calibration_size,
        shuffle=args.shuffle_calibration,
        seed=args.global_seed,
    )
    completed_sample_indices = E3.completed_indices(completed_mask)
    transition_stages = E3.transition_stage_labels(
        call_steps,
        num_distances,
        early_frac=args.early_frac,
        late_start_frac=args.late_start_frac,
    )
    call_stages = E3.call_stage_labels(
        call_steps,
        early_frac=args.early_frac,
        late_start_frac=args.late_start_frac,
    )
    forced_mask = E3.forced_call_mask(
        total_calls,
        warmup_calls=args.warmup_calls,
        force_final_call=not args.no_force_final_call,
    )

    all_rows: List[Dict[str, Any]] = []
    all_density_rows: List[Dict[str, Any]] = []
    all_stage_kind_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "meta": {
            "distance_bank": str(distance_bank_path),
            "output_dir": str(output_dir),
            "target_rrs": target_rrs,
            "num_samples": num_samples,
            "num_distances": num_distances,
            "total_calls": total_calls,
            "calibration_indices": calibration_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "warmup_calls": args.warmup_calls,
            "force_final_call": not args.no_force_final_call,
            "early_frac": args.early_frac,
            "late_start_frac": args.late_start_frac,
            "sea_log1p": args.sea_log1p,
            "normalization_eps": args.normalization_eps,
            "clip_percentile": args.clip_percentile,
        },
        "candidates": {},
    }

    for candidate_key in candidate_keys:
        label = CANDIDATE_DEFAULTS[candidate_key]["label"]
        weights = candidate_weights[candidate_key]
        score_args = SimpleNamespace(
            sea_log1p=args.sea_log1p,
            normalization_eps=args.normalization_eps,
            clip_percentile=args.clip_percentile,
            pma_nogate_weights=args.pma_nogate_weights,
            pma_stage_weights=weights,
        )
        scores, normalization = E3.build_scores(bank, calibration_indices, transition_stages, score_args)
        score = scores["pma_stageaware_oracle"]

        candidate_dir = output_dir / candidate_key
        schedule_dir = candidate_dir / "matched_schedules"
        rows: List[Dict[str, Any]] = []
        density_rows: List[Dict[str, Any]] = []
        stage_kind_rows: List[Dict[str, Any]] = []

        print(f"[E4-candidates] Building {candidate_key} ({label})", flush=True)
        for target_rr in target_rrs:
            delta, schedule, info = E3.find_threshold_for_target_rr(
                score,
                target_rr,
                calibration_indices,
                forced_mask,
                search_iters=args.threshold_search_iters,
            )
            E3.save_schedule_npz(
                schedule_dir / f"pma_stageaware_oracle_{E3.rr_label(target_rr)}.npz",
                schedule=schedule,
                method="pma_stageaware_oracle",
                target_rr=target_rr,
                selected_delta=float(delta),
                call_timesteps=call_timesteps,
                call_steps=call_steps,
                call_kinds=call_kinds,
                completed_sample_indices=completed_sample_indices,
                calibration_indices=calibration_indices,
                test_indices=test_indices,
                forced_mask=forced_mask,
                score=score,
            )
            row = E3.schedule_summary_row(
                method="pma_stageaware_oracle",
                target_rr=target_rr,
                delta=float(delta),
                schedule=schedule,
                calibration_indices=calibration_indices,
                test_indices=test_indices,
                extra={
                    "candidate": candidate_key,
                    "candidate_label": label,
                    "stage_weights": weights,
                    "target_abs_error": info.get("abs_error"),
                    "target_rr_reachable": info.get("target_rr_reachable"),
                    "min_positive_score": info.get("min_positive_score"),
                    "max_sample_score_sum": info.get("max_sample_score_sum"),
                    "schedule_path": str(schedule_dir / f"pma_stageaware_oracle_{E3.rr_label(target_rr)}.npz"),
                },
            )
            rows.append(row)
            all_rows.append(row)

            stage_density, stage_kind_density = E3.density_rows(
                method="pma_stageaware_oracle",
                target_rr=target_rr,
                schedule=schedule,
                call_stages=call_stages,
                call_kinds=call_kinds,
                calibration_indices=calibration_indices,
                test_indices=test_indices,
            )
            for density_row in stage_density:
                density_row.update({"candidate": candidate_key, "candidate_label": label, "stage_weights": weights})
            for density_row in stage_kind_density:
                density_row.update({"candidate": candidate_key, "candidate_label": label, "stage_weights": weights})
            density_rows.extend(stage_density)
            stage_kind_rows.extend(stage_kind_density)
            all_density_rows.extend(stage_density)
            all_stage_kind_rows.extend(stage_kind_density)

            print(
                f"[E4-candidates]   RR={target_rr:.2f} delta={float(delta):.6g} "
                f"test_rr={row['test_rr']:.4f} all_rr={row['all_rr']:.4f}",
                flush=True,
            )

        fields = list(rows[0].keys()) if rows else []
        write_csv(candidate_dir / "schedule_summary.csv", rows, fields)
        write_csv(candidate_dir / "stage_refresh_density.csv", density_rows)
        write_csv(candidate_dir / "stage_kind_refresh_density.csv", stage_kind_rows)
        with open(candidate_dir / "candidate_summary.json", "w", encoding="utf-8") as fp:
            json.dump(
                json_ready(
                    {
                        "candidate": candidate_key,
                        "candidate_label": label,
                        "stage_weights": weights,
                        "schedule_dir": schedule_dir,
                        "normalization": normalization,
                        "schedule_rows": rows,
                    }
                ),
                fp,
                indent=2,
            )
        summary["candidates"][candidate_key] = {
            "label": label,
            "stage_weights": weights,
            "schedule_dir": str(schedule_dir),
            "schedule_rows": rows,
        }

    if all_rows:
        write_csv(output_dir / "schedule_summary.csv", all_rows, list(all_rows[0].keys()))
    write_csv(output_dir / "stage_refresh_density.csv", all_density_rows)
    write_csv(output_dir / "stage_kind_refresh_density.csv", all_stage_kind_rows)
    summary["elapsed_sec"] = time.perf_counter() - start
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(json_ready(summary), fp, indent=2)

    print(f"[E4-candidates] Wrote candidate schedules to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
