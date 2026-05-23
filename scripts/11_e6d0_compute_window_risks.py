#!/usr/bin/env python3
"""Compute E6-D0 window risk scores from full trajectory dumps."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cache.xwp.frequency import fft_filter_2d, radial_frequency_grid  # noqa: E402
from src.cache.xwp.risk_scores import (  # noqa: E402
    ode_factor,
    posterior_uncertainty_xpred,
    scalar_window_risk,
    symmetric_relative_l1,
    vector_window_risk,
)
from src.cache.xwp.perceptual_weight import phi_perceptual  # noqa: E402
from src.cache.xwp.wiener_proxy import wiener_clean_proxy, wiener_filter_xpred  # noqa: E402


BASE_LABEL_COLUMNS = [
    "sample_id",
    "sample_index",
    "class_id",
    "seed",
    "window_id",
    "stage",
    "start_call",
    "end_call",
    "window_len",
    "pis_lpips",
    "pis_dino",
    "pis_total_z",
    "pis_total_rank",
]

RISK_COLUMNS = [
    "R_time_len",
    "R_time_sum_ode",
    "R_time_mean_t",
    "R_time_inv_1mt",
    "R_raw_adj",
    "R_raw_anchor",
    "R_sea_adj",
    "R_sea_anchor",
    "R_xw_adj",
    "R_xw_anchor",
    "R_xwp_anchor",
    "R_xwp_ode_scalar",
    "R_xwp_ode_vector",
    "R_uncertainty_sum",
    "R_uncertainty_mean",
    "R_xwp_ode_scalar_u_eta0p03",
    "R_xwp_ode_scalar_u_eta0p1",
    "R_xwp_ode_scalar_u_eta0p3",
    "R_xwp_ode_vector_u_eta0p03",
    "R_xwp_ode_vector_u_eta0p1",
    "R_xwp_ode_vector_u_eta0p3",
    "R_oracle_xhat_scalar",
    "R_oracle_xhat_vector",
    "R_oracle_xhat_scalar_u_eta0p1",
    "R_oracle_xhat_vector_u_eta0p1",
]


def resolve_path(path: str | Path) -> Path:
    item = Path(path).expanduser()
    if item.is_absolute():
        return item
    return (PROJECT_ROOT / item).resolve()


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def eta_name(value: float) -> str:
    text = f"{value:g}".replace(".", "p").replace("-", "m")
    return text


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def as_bchw(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected tensor shape [C,H,W] or [B,C,H,W], got {tuple(x.shape)}")
    return x.to(device=device, dtype=torch.float32)


def downsample(x: torch.Tensor, size: int, device: torch.device) -> torch.Tensor:
    x_b = as_bchw(x, device)
    if x_b.shape[-2:] != (int(size), int(size)):
        x_b = F.interpolate(x_b, size=(int(size), int(size)), mode="bilinear", align_corners=False)
    return x_b


def rel_l1_float(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(symmetric_relative_l1(a, b, eps=1e-8).detach().cpu())


def sea_xt_proxy(x_t: torch.Tensor, t: float, size: int, beta: float, f0: float, device: torch.device) -> torch.Tensor:
    x_small = downsample(x_t, size=size, device=device)
    freq = radial_frequency_grid(size, size, device=device, dtype=torch.float32, normalize=True)
    filt = wiener_filter_xpred(t, freq, beta=beta, f0=f0, eps=1e-8, normalize_mean=True)
    return fft_filter_2d(x_small, filt, norm="ortho")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        item = float(value)
    except Exception:
        return default
    return item if math.isfinite(item) else default


def zero_risks() -> Dict[str, float]:
    return {key: 0.0 for key in RISK_COLUMNS}


def finite_or_raise(scores: Mapping[str, Any]) -> None:
    bad = [key for key, value in scores.items() if key.startswith("R_") and not math.isfinite(float(value))]
    if bad:
        raise RuntimeError(f"Non-finite risk scores: {bad}")


def compute_adj_sum(values: Sequence[torch.Tensor]) -> float:
    return float(sum(rel_l1_float(values[idx], values[idx - 1]) for idx in range(1, len(values))))


def compute_window_scores(
    *,
    label: Mapping[str, Any],
    traj: Mapping[str, Any],
    device: torch.device,
    proxy_size: int,
    beta: float,
    f0: float,
    epsilon_clip: float,
    eta_list: Sequence[float],
) -> Dict[str, Any]:
    calls = traj.get("calls")
    if not isinstance(calls, list):
        raise ValueError("Trajectory payload has no calls list")
    if len(calls) != 99:
        raise ValueError(f"Expected 99 calls, got {len(calls)}")

    start_call = int(label["start_call"])
    end_call = int(label["end_call"])
    anchor_idx = start_call - 1
    if anchor_idx < 0:
        raise ValueError("anchor<0")
    if end_call >= len(calls):
        raise ValueError(f"window end_call {end_call} outside trajectory with {len(calls)} calls")

    window_calls = calls[start_call : end_call + 1]
    anchor_call = calls[anchor_idx]
    anchor_xhat = anchor_call["xhat"]
    anchor_xt = anchor_call["x_t"]

    t_values = np.asarray([float(call["t"]) for call in window_calls], dtype=np.float64)
    h_values = np.asarray([float(call.get("effective_solver_coeff", call.get("h", 0.0))) for call in window_calls])
    ode_values = np.asarray([float(ode_factor(t, h, epsilon_clip=epsilon_clip)) for t, h in zip(t_values, h_values)])
    kinds = [str(call["call_kind"]) for call in window_calls]

    scores: Dict[str, Any] = zero_risks()
    scores.update(
        {
            "mean_t": float(t_values.mean()),
            "min_t": float(t_values.min()),
            "max_t": float(t_values.max()),
            "num_predictor": int(sum(kind == "predictor" for kind in kinds)),
            "num_corrector": int(sum(kind == "corrector" for kind in kinds)),
        }
    )
    scores["R_time_len"] = float(len(window_calls))
    scores["R_time_sum_ode"] = float(ode_values.sum())
    scores["R_time_mean_t"] = float(t_values.mean())
    scores["R_time_inv_1mt"] = float(sum(1.0 / max(1.0 - float(t), epsilon_clip) for t in t_values))

    x_anchor_down = downsample(anchor_xt, proxy_size, device)
    x_window_down = [downsample(call["x_t"], proxy_size, device) for call in window_calls]
    x_prev_down = [downsample(calls[idx - 1]["x_t"], proxy_size, device) for idx in range(start_call, end_call + 1)]
    scores["R_raw_adj"] = float(sum(rel_l1_float(cur, prev) for cur, prev in zip(x_window_down, x_prev_down)))
    scores["R_raw_anchor"] = float(sum(rel_l1_float(cur, x_anchor_down) for cur in x_window_down))

    sea_window = [sea_xt_proxy(call["x_t"], float(call["t"]), proxy_size, beta, f0, device) for call in window_calls]
    sea_prev = [sea_xt_proxy(calls[idx - 1]["x_t"], float(calls[idx - 1]["t"]), proxy_size, beta, f0, device) for idx in range(start_call, end_call + 1)]
    sea_anchor_by_t = [sea_xt_proxy(anchor_xt, float(call["t"]), proxy_size, beta, f0, device) for call in window_calls]
    scores["R_sea_adj"] = float(sum(rel_l1_float(cur, prev) for cur, prev in zip(sea_window, sea_prev)))
    scores["R_sea_anchor"] = float(sum(rel_l1_float(cur, anc) for cur, anc in zip(sea_window, sea_anchor_by_t)))

    xw_window = [
        wiener_clean_proxy(as_bchw(call["x_t"], device), float(call["t"]), size=proxy_size, beta=beta, f0=f0)
        for call in window_calls
    ]
    xw_prev = [
        wiener_clean_proxy(as_bchw(calls[idx - 1]["x_t"], device), float(calls[idx - 1]["t"]), size=proxy_size, beta=beta, f0=f0)
        for idx in range(start_call, end_call + 1)
    ]
    xhat_anchor_down = downsample(anchor_xhat, proxy_size, device)
    scores["R_xw_adj"] = float(sum(rel_l1_float(cur, prev) for cur, prev in zip(xw_window, xw_prev)))
    scores["R_xw_anchor"] = float(sum(rel_l1_float(cur, xhat_anchor_down) for cur in xw_window))

    z_dists = []
    for call, xw in zip(window_calls, xw_window):
        t = float(call["t"])
        z_cur = phi_perceptual(xw, t, size=proxy_size)
        z_anchor = phi_perceptual(as_bchw(anchor_xhat, device), t, size=proxy_size)
        z_dists.append(rel_l1_float(z_anchor, z_cur))
    scores["R_xwp_anchor"] = float(sum(z_dists))

    scores["R_xwp_ode_scalar"] = float(
        scalar_window_risk(
            window_calls,
            anchor_xhat,
            proxy_mode="wiener",
            use_perceptual=True,
            use_ode=True,
            size=proxy_size,
            beta=beta,
            f0=f0,
            epsilon_clip=epsilon_clip,
            device=device,
        )
    )
    vector = vector_window_risk(
        window_calls,
        anchor_xhat,
        proxy_mode="wiener",
        use_perceptual=True,
        use_ode=True,
        size=proxy_size,
        beta=beta,
        f0=f0,
        epsilon_clip=epsilon_clip,
        device=device,
    )
    scores["R_xwp_ode_vector"] = float(vector["risk"])

    uncertainties = [
        float(posterior_uncertainty_xpred(float(t), size=proxy_size, beta=beta, f0=f0, perceptual_weight=True).detach().cpu())
        for t in t_values
    ]
    scores["R_uncertainty_sum"] = float(sum(uncertainties))
    scores["R_uncertainty_mean"] = float(np.mean(uncertainties))

    for eta in eta_list:
        if abs(float(eta)) <= 1e-12:
            continue
        name = eta_name(float(eta))
        scalar_name = f"R_xwp_ode_scalar_u_eta{name}"
        vector_name = f"R_xwp_ode_vector_u_eta{name}"
        if scalar_name in scores:
            scores[scalar_name] = float(scores["R_xwp_ode_scalar"] + float(eta) * scores["R_uncertainty_sum"])
        if vector_name in scores:
            scores[vector_name] = float(scores["R_xwp_ode_vector"] + float(eta) * scores["R_uncertainty_sum"])

    scores["R_oracle_xhat_scalar"] = float(
        scalar_window_risk(
            window_calls,
            anchor_xhat,
            proxy_mode="oracle_xhat",
            use_perceptual=True,
            use_ode=True,
            size=proxy_size,
            beta=beta,
            f0=f0,
            epsilon_clip=epsilon_clip,
            device=device,
        )
    )
    oracle_vector = vector_window_risk(
        window_calls,
        anchor_xhat,
        proxy_mode="oracle_xhat",
        use_perceptual=True,
        use_ode=True,
        size=proxy_size,
        beta=beta,
        f0=f0,
        epsilon_clip=epsilon_clip,
        device=device,
    )
    scores["R_oracle_xhat_vector"] = float(oracle_vector["risk"])
    scores["R_oracle_xhat_scalar_u_eta0p1"] = float(scores["R_oracle_xhat_scalar"] + 0.1 * scores["R_uncertainty_sum"])
    scores["R_oracle_xhat_vector_u_eta0p1"] = float(scores["R_oracle_xhat_vector"] + 0.1 * scores["R_uncertainty_sum"])
    scores["_vector_details"] = vector
    scores["_oracle_vector_details"] = oracle_vector
    finite_or_raise(scores)
    return scores


def output_row_from_label(label: Mapping[str, Any]) -> Dict[str, Any]:
    row = {key: label.get(key, "") for key in BASE_LABEL_COLUMNS}
    for key in ["pis_psnr", "pis_ssim", "start_step", "end_step", "start_t", "end_t", "start_call_kind", "end_call_kind"]:
        if key in label:
            row[key] = label.get(key)
    row.update({"mean_t": 0.0, "min_t": 0.0, "max_t": 0.0, "num_predictor": 0, "num_corrector": 0})
    row.update(zero_risks())
    row.update({"valid_risk": False, "invalid_reason": "", "n_calls": 0, "risk_compute_sec": 0.0})
    return row


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="outputs/e6_d0_labels/e55_window_labels.csv")
    parser.add_argument("--fulltraj", default="outputs/e6_d0_fulltraj/main8")
    parser.add_argument("--out", default="outputs/e6_d0_xwp_diagnostic/main8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--proxy-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--f0", type=float, default=0.03)
    parser.add_argument("--epsilon-clip", type=float, default=0.05)
    parser.add_argument("--eta-list", default="0,0.03,0.1,0.3")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--score-subset", default=None, help="Reserved for narrow debug runs; all scores are computed by default.")
    parser.add_argument("--save-details", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    labels_path = resolve_path(args.labels)
    fulltraj_dir = resolve_path(args.fulltraj)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_dir = out_dir / "details"
    if args.save_details:
        details_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[E6-D0 risks] CUDA requested but unavailable; falling back to CPU", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    eta_list = parse_float_list(args.eta_list)

    labels = pd.read_csv(labels_path)
    if args.max_rows is not None:
        labels = labels.head(int(args.max_rows)).copy()

    traj_cache: Dict[int, Mapping[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for row_idx, label_row in labels.iterrows():
        label = label_row.to_dict()
        out_row = output_row_from_label(label)
        row_start = time.perf_counter()
        sample_index = int(label["sample_index"])
        try:
            if sample_index not in traj_cache:
                sample_path = fulltraj_dir / f"sample_{sample_index:03d}.pt"
                if not sample_path.exists():
                    raise FileNotFoundError(f"missing trajectory file: {sample_path}")
                traj_cache[sample_index] = torch.load(sample_path, map_location="cpu", weights_only=False)
            scores = compute_window_scores(
                label=label,
                traj=traj_cache[sample_index],
                device=device,
                proxy_size=int(args.proxy_size),
                beta=float(args.beta),
                f0=float(args.f0),
                epsilon_clip=float(args.epsilon_clip),
                eta_list=eta_list,
            )
            details = {
                "_vector_details": scores.pop("_vector_details", {}),
                "_oracle_vector_details": scores.pop("_oracle_vector_details", {}),
            }
            out_row.update(scores)
            out_row["valid_risk"] = True
            out_row["invalid_reason"] = ""
            out_row["n_calls"] = int(out_row["window_len"])
            if args.save_details:
                detail_path = details_dir / f"row{row_idx:05d}_sample{sample_index:03d}_window{int(label['window_id']):03d}.json"
                with open(detail_path, "w", encoding="utf-8") as fp:
                    json.dump(json_ready(details), fp, indent=2)
        except Exception as exc:
            out_row["valid_risk"] = False
            out_row["invalid_reason"] = str(exc)
        out_row["risk_compute_sec"] = time.perf_counter() - row_start
        for key in RISK_COLUMNS:
            out_row[key] = safe_float(out_row.get(key), 0.0)
        rows.append(out_row)
        if (len(rows) % 25) == 0:
            print(f"[E6-D0 risks] processed {len(rows)}/{len(labels)} rows", flush=True)

    table = pd.DataFrame(rows)
    risk_path = out_dir / "window_risk_table.csv"
    table.to_csv(risk_path, index=False)

    score_stats: Dict[str, Dict[str, float]] = {}
    for col in RISK_COLUMNS:
        values = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            score_stats[col] = {"min": float(finite.min()), "mean": float(finite.mean()), "max": float(finite.max())}
        else:
            score_stats[col] = {"min": 0.0, "mean": 0.0, "max": 0.0}

    config = {
        "labels": str(labels_path),
        "fulltraj": str(fulltraj_dir),
        "out": str(out_dir),
        "device": str(device),
        "proxy_size": int(args.proxy_size),
        "beta": float(args.beta),
        "f0": float(args.f0),
        "epsilon_clip": float(args.epsilon_clip),
        "eta_list": eta_list,
        "score_subset": args.score_subset,
    }
    with open(out_dir / "risk_config.json", "w", encoding="utf-8") as fp:
        json.dump(json_ready(config), fp, indent=2)

    summary = {
        **config,
        "risk_table": str(risk_path),
        "num_label_rows": int(len(labels)),
        "num_output_rows": int(len(table)),
        "num_valid_risk": int(table["valid_risk"].astype(bool).sum()),
        "num_invalid_risk": int((~table["valid_risk"].astype(bool)).sum()),
        "invalid_reason_counts": table.loc[~table["valid_risk"].astype(bool), "invalid_reason"].value_counts().to_dict(),
        "score_stats": score_stats,
        "elapsed_sec": time.perf_counter() - started,
    }
    with open(out_dir / "compute_summary.json", "w", encoding="utf-8") as fp:
        json.dump(json_ready(summary), fp, indent=2)

    print(f"[E6-D0 risks] wrote {risk_path}", flush=True)
    print("[E6-D0 risks] score min/mean/max:", flush=True)
    for key in RISK_COLUMNS:
        stat = score_stats[key]
        print(f"  {key}: {stat['min']:.6g} / {stat['mean']:.6g} / {stat['max']:.6g}", flush=True)


if __name__ == "__main__":
    main()
