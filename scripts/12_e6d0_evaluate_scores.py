#!/usr/bin/env python3
"""Evaluate E6-D0 risk scores against E5.5 window PIS labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TARGET_COLUMNS = ["pis_lpips", "pis_dino", "pis_total_z", "pis_total_rank"]
DANGER_COLUMNS = {
    "dangerous_total": "pis_total_rank",
    "dangerous_lpips": "pis_lpips",
    "dangerous_dino": "pis_dino",
}
TIME_FEATURES = [
    "start_call",
    "end_call",
    "window_len",
    "mean_t",
    "min_t",
    "max_t",
    "num_predictor",
    "num_corrector",
]
MAIN_COMPARE_SCORES = [
    "R_time_len",
    "R_time_sum_ode",
    "R_raw_anchor",
    "R_sea_anchor",
    "R_xw_anchor",
    "R_xwp_anchor",
    "R_xwp_ode_scalar",
    "R_xwp_ode_vector",
    "R_xwp_ode_scalar_u_eta0p1",
    "R_xwp_ode_vector_u_eta0p1",
    "R_oracle_xhat_scalar",
    "R_oracle_xhat_vector",
]


def resolve_path(path: str | Path) -> Path:
    item = Path(path).expanduser()
    if item.is_absolute():
        return item
    return (PROJECT_ROOT / item).resolve()


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


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


def spearman_corr(x: Iterable[Any], y: Iterable[Any]) -> float:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3:
        return float("nan")
    x_use = x_arr[mask]
    y_use = y_arr[mask]
    if np.all(x_use == x_use[0]) or np.all(y_use == y_use[0]):
        return float("nan")
    rx = rankdata_average(x_use)
    ry = rankdata_average(y_use)
    corr = np.corrcoef(rx, ry)[0, 1]
    return float(corr)


def roc_auc_score_fallback(y_true: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    y = y_true[mask].astype(bool)
    s = scores[mask].astype(np.float64)
    pos = int(y.sum())
    neg = int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata_average(s)
    rank_sum_pos = float(ranks[y].sum())
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc)


def pr_auc_score_fallback(y_true: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    y = y_true[mask].astype(bool)
    s = scores[mask].astype(np.float64)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(y_sorted.size) + 1.0)
    return float(precision[y_sorted].sum() / pos)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        mask = np.isfinite(scores)
        if int(y_true[mask].sum()) == 0 or int((~y_true[mask].astype(bool)).sum()) == 0:
            return float("nan")
        return float(roc_auc_score(y_true[mask].astype(int), scores[mask]))
    except Exception:
        return roc_auc_score_fallback(y_true, scores)


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score  # type: ignore

        mask = np.isfinite(scores)
        if int(y_true[mask].sum()) == 0:
            return float("nan")
        return float(average_precision_score(y_true[mask].astype(int), scores[mask]))
    except Exception:
        return pr_auc_score_fallback(y_true, scores)


def dangerous_mask(values: np.ndarray, top_frac: float) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=bool)
    cutoff = float(np.quantile(finite, 1.0 - float(top_frac)))
    return np.isfinite(values) & (values >= cutoff)


def top_budget_mask(scores: np.ndarray, budget: float) -> np.ndarray:
    mask = np.isfinite(scores)
    out = np.zeros(scores.shape, dtype=bool)
    n = int(mask.sum())
    if n <= 0 or budget <= 0:
        return out
    k = max(1, int(math.ceil(float(budget) * n)))
    finite_indices = np.flatnonzero(mask)
    order = finite_indices[np.argsort(-scores[finite_indices], kind="mergesort")]
    out[order[:k]] = True
    return out


def captured_pis(values: np.ndarray, selected: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(values)
    denom = float(values[mask].sum())
    if denom <= 0:
        return float("nan"), float("nan")
    captured = float(values[mask & selected].sum() / denom)
    return captured, 1.0 - captured


def rank_percentile_high(scores: np.ndarray) -> np.ndarray:
    ranks = rankdata_average(scores.astype(np.float64))
    if scores.size <= 1:
        return np.ones_like(scores, dtype=np.float64)
    return (ranks - 1.0) / float(scores.size - 1)


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def fit_ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    x_train_s, x_test_s = standardize_train_test(x_train, x_test)
    x_design = np.concatenate([np.ones((x_train_s.shape[0], 1)), x_train_s], axis=1)
    x_test_design = np.concatenate([np.ones((x_test_s.shape[0], 1)), x_test_s], axis=1)
    reg = alpha * np.eye(x_design.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    try:
        coef = np.linalg.solve(x_design.T @ x_design + reg, x_design.T @ y_train)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(x_design.T @ x_design + reg, x_design.T @ y_train, rcond=None)[0]
    return x_test_design @ coef


def time_residuals(df: pd.DataFrame, target: str) -> Dict[str, np.ndarray]:
    x = df[TIME_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    residual_in = np.full(df.shape[0], np.nan, dtype=np.float64)
    residual_loso = np.full(df.shape[0], np.nan, dtype=np.float64)
    if int(valid.sum()) >= len(TIME_FEATURES) + 2:
        pred = fit_ridge_predict(x[valid], y[valid], x[valid])
        residual_in[valid] = y[valid] - pred
    samples = pd.to_numeric(df["sample_index"], errors="coerce").to_numpy()
    for sample in sorted(set(samples[valid].tolist())):
        test = valid & (samples == sample)
        train = valid & (samples != sample)
        if int(train.sum()) < len(TIME_FEATURES) + 2 or int(test.sum()) == 0:
            continue
        pred = fit_ridge_predict(x[train], y[train], x[test])
        residual_loso[test] = y[test] - pred
    return {"insample": residual_in, "leave_one_sample_out": residual_loso}


def write_empty_outputs(out_dir: Path, reason: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "score_metrics.csv",
        "score_metrics_by_window_len.csv",
        "score_metrics_by_stage.csv",
        "threshold_sweep.csv",
        "captured_pis_at_budget.csv",
        "false_negative_at_budget.csv",
        "controlled_correlation.csv",
        "failure_false_negative.csv",
    ]:
        pd.DataFrame().to_csv(out_dir / name, index=False)
    summary = {"status": "empty", "reason": reason}
    with open(out_dir / "best_scores_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    (out_dir / "report.md").write_text(f"# E6-D0 Report\n\nNo valid rows to evaluate: {reason}\n", encoding="utf-8")


def maybe_plot(out_dir: Path, plot_func) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_func(plt)
    except Exception as exc:
        print(f"[E6-D0 eval] plot skipped: {exc}", flush=True)


def bar_plot(path: Path, data: pd.DataFrame, value_col: str, title: str, ylabel: str) -> None:
    def _plot(plt):
        plot_df = data[["score", value_col]].dropna().sort_values(value_col, ascending=False).head(20)
        if plot_df.empty:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(max(8, 0.45 * len(plot_df)), 4.5))
        plt.bar(range(len(plot_df)), plot_df[value_col].to_numpy())
        plt.xticks(range(len(plot_df)), plot_df["score"].tolist(), rotation=70, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    maybe_plot(path.parent, _plot)


def curve_plot(path: Path, rows: pd.DataFrame, scores: Sequence[str], value_col: str, title: str, ylabel: str) -> None:
    def _plot(plt):
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 4.5))
        for score in scores:
            subset = rows[(rows["score"] == score) & (rows["target"] == "pis_total_rank")].sort_values("budget")
            if subset.empty:
                continue
            plt.plot(subset["budget"], subset[value_col], marker="o", label=score)
        plt.xlabel("refresh budget fraction")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    maybe_plot(path.parent, _plot)


def scatter_plot(path: Path, df: pd.DataFrame, score: str) -> None:
    def _plot(plt):
        x = pd.to_numeric(df[score], errors="coerce")
        y = pd.to_numeric(df["pis_total_rank"], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 3:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(5, 4.5))
        plt.scatter(x[mask], y[mask], s=14, alpha=0.7)
        plt.xlabel(score)
        plt.ylabel("pis_total_rank")
        plt.title(f"PIS vs {score}")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    maybe_plot(path.parent, _plot)


def heatmap_plot(path: Path, df: pd.DataFrame, score: str) -> None:
    def _plot(plt):
        table = (
            df.pivot_table(index="window_len", columns="start_call", values=score, aggfunc="mean")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        if table.empty:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(max(7, 0.35 * table.shape[1]), max(3.5, 0.45 * table.shape[0])))
        plt.imshow(table.to_numpy(dtype=np.float64), aspect="auto")
        plt.colorbar(label=f"mean {score}")
        plt.xticks(range(table.shape[1]), table.columns.tolist(), rotation=90)
        plt.yticks(range(table.shape[0]), table.index.tolist())
        plt.xlabel("start_call")
        plt.ylabel("window_len")
        plt.title(f"Mean risk heatmap: {score}")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    maybe_plot(path.parent, _plot)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-table", default="outputs/e6_d0_xwp_diagnostic/main8/window_risk_table.csv")
    parser.add_argument("--out", default="outputs/e6_d0_xwp_diagnostic/main8/eval")
    parser.add_argument("--danger-top-frac", type=float, default=0.2)
    parser.add_argument("--budgets", default="0.2,0.3,0.4,0.5")
    parser.add_argument("--score-columns", default=None, help="Comma-separated score columns. Defaults to all R_* columns.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    risk_table = resolve_path(args.risk_table)
    out_dir = resolve_path(args.out)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    budgets = parse_float_list(args.budgets)

    df_all = pd.read_csv(risk_table)
    if "valid_risk" in df_all.columns:
        valid_mask = df_all["valid_risk"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
        df = df_all.loc[valid_mask].copy()
    else:
        df = df_all.copy()
    if df.empty:
        write_empty_outputs(out_dir, "risk table has no valid_risk rows")
        return

    for col in TARGET_COLUMNS + TIME_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    score_columns = parse_str_list(args.score_columns)
    if score_columns is None:
        score_columns = [col for col in df.columns if col.startswith("R_")]
    score_columns = [col for col in score_columns if col in df.columns]
    if not score_columns:
        write_empty_outputs(out_dir, "no score columns found")
        return
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    danger_masks = {
        name: dangerous_mask(pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=np.float64), args.danger_top_frac)
        for name, target in DANGER_COLUMNS.items()
    }
    residuals = time_residuals(df, "pis_total_rank")

    metrics_rows: List[Dict[str, Any]] = []
    by_len_rows: List[Dict[str, Any]] = []
    by_stage_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    captured_rows: List[Dict[str, Any]] = []
    fnr_rows: List[Dict[str, Any]] = []
    controlled_rows: List[Dict[str, Any]] = []

    for score in score_columns:
        score_values = df[score].to_numpy(dtype=np.float64)
        row: Dict[str, Any] = {"score": score, "n": int(np.isfinite(score_values).sum())}
        for target in TARGET_COLUMNS:
            row[f"spearman_{target}"] = spearman_corr(score_values, df[target].to_numpy(dtype=np.float64))
        for danger_name, mask in danger_masks.items():
            row[f"roc_auc_{danger_name}"] = roc_auc(mask, score_values)
            row[f"pr_auc_{danger_name}"] = pr_auc(mask, score_values)
        for mode, residual in residuals.items():
            value = spearman_corr(score_values, residual)
            row[f"controlled_spearman_pis_total_rank_{mode}"] = value
            controlled_rows.append({"score": score, "mode": mode, "target": "pis_total_rank", "spearman_residual": value})

        for budget in budgets:
            selected = top_budget_mask(score_values, budget)
            threshold = float(np.nanmin(score_values[selected])) if selected.any() else float("nan")
            sweep_base = {
                "score": score,
                "budget": float(budget),
                "threshold": threshold,
                "selected_count": int(selected.sum()),
                "total_count": int(np.isfinite(score_values).sum()),
            }
            for target in ["pis_total_rank", "pis_lpips", "pis_dino"]:
                values = df[target].to_numpy(dtype=np.float64)
                captured, skipped = captured_pis(values, selected)
                captured_rows.append(
                    {
                        **sweep_base,
                        "target": target,
                        "captured_pis": captured,
                        "skipped_pis": skipped,
                    }
                )
                if target == "pis_total_rank" and abs(float(budget) - 0.3) < 1e-9:
                    row["captured_pis_total_rank_at_0p3"] = captured
                    row["skipped_pis_total_rank_at_0p3"] = skipped
            for danger_name, mask in danger_masks.items():
                denom = int(mask.sum())
                fnr = float((mask & ~selected).sum() / denom) if denom > 0 else float("nan")
                fnr_rows.append({**sweep_base, "danger_target": danger_name, "fnr": fnr})
                if danger_name == "dangerous_total" and abs(float(budget) - 0.3) < 1e-9:
                    row["fnr_dangerous_total_at_0p3"] = fnr
            threshold_rows.append(sweep_base)
        metrics_rows.append(row)

        for length, sub in df.groupby("window_len"):
            out = {"score": score, "window_len": int(length), "n": int(len(sub))}
            for target in TARGET_COLUMNS:
                out[f"spearman_{target}"] = spearman_corr(sub[score], sub[target])
            by_len_rows.append(out)
        for stage, sub in df.groupby("stage"):
            out = {"score": score, "stage": str(stage), "n": int(len(sub))}
            for target in TARGET_COLUMNS:
                out[f"spearman_{target}"] = spearman_corr(sub[score], sub[target])
            by_stage_rows.append(out)

    metrics = pd.DataFrame(metrics_rows)
    by_len = pd.DataFrame(by_len_rows)
    by_stage = pd.DataFrame(by_stage_rows)
    threshold_sweep = pd.DataFrame(threshold_rows)
    captured_df = pd.DataFrame(captured_rows)
    fnr_df = pd.DataFrame(fnr_rows)
    controlled_df = pd.DataFrame(controlled_rows)

    metrics.to_csv(out_dir / "score_metrics.csv", index=False)
    by_len.to_csv(out_dir / "score_metrics_by_window_len.csv", index=False)
    by_stage.to_csv(out_dir / "score_metrics_by_stage.csv", index=False)
    threshold_sweep.to_csv(out_dir / "threshold_sweep.csv", index=False)
    captured_df.to_csv(out_dir / "captured_pis_at_budget.csv", index=False)
    fnr_df.to_csv(out_dir / "false_negative_at_budget.csv", index=False)
    controlled_df.to_csv(out_dir / "controlled_correlation.csv", index=False)

    failure_scores = [score for score in ["R_xwp_ode_vector", "R_xwp_ode_vector_u_eta0p1", "R_oracle_xhat_vector"] if score in df]
    failure_rows: List[Dict[str, Any]] = []
    danger_total = danger_masks["dangerous_total"]
    for score in failure_scores:
        scores = df[score].to_numpy(dtype=np.float64)
        selected30 = top_budget_mask(scores, 0.3)
        percentiles = rank_percentile_high(np.nan_to_num(scores, nan=-np.inf))
        fail_mask = danger_total & ~selected30
        for idx in np.flatnonzero(fail_mask):
            item = df.iloc[idx]
            failure_rows.append(
                {
                    "score_column": score,
                    "sample_id": int(item.get("sample_id", item.get("sample_index", -1))),
                    "sample_index": int(item.get("sample_index", -1)),
                    "window_id": int(item.get("window_id", -1)),
                    "start_call": int(item.get("start_call", -1)),
                    "end_call": int(item.get("end_call", -1)),
                    "window_len": int(item.get("window_len", -1)),
                    "stage": item.get("stage", ""),
                    "pis_total_rank": float(item.get("pis_total_rank", float("nan"))),
                    "pis_lpips": float(item.get("pis_lpips", float("nan"))),
                    "pis_dino": float(item.get("pis_dino", float("nan"))),
                    "score_value": float(item.get(score, float("nan"))),
                    "rank_percentile": float(percentiles[idx]),
                }
            )
    failure_df = pd.DataFrame(failure_rows)
    failure_df.to_csv(out_dir / "failure_false_negative.csv", index=False)

    best_spearman = metrics.dropna(subset=["spearman_pis_total_rank"]).sort_values(
        "spearman_pis_total_rank", ascending=False
    )
    best_pr = metrics.dropna(subset=["pr_auc_dangerous_total"]).sort_values("pr_auc_dangerous_total", ascending=False)
    best_captured = metrics.dropna(subset=["captured_pis_total_rank_at_0p3"]).sort_values(
        "captured_pis_total_rank_at_0p3", ascending=False
    )

    top_plot_scores = (
        best_pr["score"].head(8).tolist()
        if not best_pr.empty
        else metrics.sort_values("spearman_pis_total_rank", ascending=False)["score"].head(8).tolist()
    )
    bar_plot(plots_dir / "spearman_bar.png", metrics, "spearman_pis_total_rank", "Spearman vs PIS total rank", "Spearman")
    bar_plot(plots_dir / "pr_auc_bar.png", metrics, "pr_auc_dangerous_total", "Dangerous total PR-AUC", "PR-AUC")
    curve_plot(plots_dir / "captured_pis_curve.png", captured_df, top_plot_scores, "captured_pis", "Captured PIS@budget", "Captured PIS")
    curve_plot(plots_dir / "skipped_pis_curve.png", captured_df, top_plot_scores, "skipped_pis", "Skipped PIS@budget", "Skipped PIS")
    for score in top_plot_scores:
        scatter_plot(plots_dir / f"scatter_pis_vs_{score}.png", df, score)
        heatmap_plot(plots_dir / f"heatmap_{score}.png", df, score)

    def metric_value(score: str, col: str) -> Optional[float]:
        if score not in set(metrics["score"]):
            return None
        value = metrics.loc[metrics["score"] == score, col].iloc[0]
        return None if pd.isna(value) else float(value)

    baselines = [score for score in ["R_time_sum_ode", "R_time_len", "R_raw_anchor", "R_sea_anchor"] if score in set(metrics["score"])]
    xwp_main = "R_xwp_ode_vector_u_eta0p1" if "R_xwp_ode_vector_u_eta0p1" in set(metrics["score"]) else "R_xwp_ode_vector"
    baseline_best_spear = max([metric_value(score, "spearman_pis_total_rank") or -math.inf for score in baselines], default=-math.inf)
    baseline_best_pr = max([metric_value(score, "pr_auc_dangerous_total") or -math.inf for score in baselines], default=-math.inf)
    baseline_best_captured = max(
        [metric_value(score, "captured_pis_total_rank_at_0p3") or -math.inf for score in baselines],
        default=-math.inf,
    )
    xwp_spear = metric_value(xwp_main, "spearman_pis_total_rank")
    xwp_pr = metric_value(xwp_main, "pr_auc_dangerous_total")
    xwp_captured = metric_value(xwp_main, "captured_pis_total_rank_at_0p3")
    xwp_controlled = metric_value(xwp_main, "controlled_spearman_pis_total_rank_leave_one_sample_out")
    xwp_beats_baseline_spearman = xwp_spear is not None and xwp_spear > baseline_best_spear
    xwp_beats_baseline_pr = xwp_pr is not None and xwp_pr > baseline_best_pr
    xwp_beats_baseline_captured = xwp_captured is not None and xwp_captured > baseline_best_captured

    vector_spear = metric_value("R_xwp_ode_vector", "spearman_pis_total_rank")
    scalar_spear = metric_value("R_xwp_ode_scalar", "spearman_pis_total_rank")
    uncertainty_spear = metric_value("R_xwp_ode_vector_u_eta0p1", "spearman_pis_total_rank")
    oracle_spear = metric_value("R_oracle_xhat_vector", "spearman_pis_total_rank")
    wiener_spear = metric_value("R_xwp_ode_vector", "spearman_pis_total_rank")

    summary = {
        "risk_table": str(risk_table),
        "out_dir": str(out_dir),
        "num_valid_rows": int(len(df)),
        "danger_top_frac": float(args.danger_top_frac),
        "budgets": budgets,
        "best_spearman_pis_total_rank": None if best_spearman.empty else best_spearman.iloc[0].to_dict(),
        "best_pr_auc_dangerous_total": None if best_pr.empty else best_pr.iloc[0].to_dict(),
        "best_captured_pis_total_rank_at_0p3": None if best_captured.empty else best_captured.iloc[0].to_dict(),
        "xwp_main_score": xwp_main,
        "xwp_beats_time_raw_sea_by_spearman": bool(xwp_beats_baseline_spearman),
        "xwp_beats_time_raw_sea_by_pr_auc": bool(xwp_beats_baseline_pr),
        "xwp_beats_time_raw_sea_by_captured_pis_at_0p3": bool(xwp_beats_baseline_captured),
        "top_plot_scores": top_plot_scores,
    }
    with open(out_dir / "best_scores_summary.json", "w", encoding="utf-8") as fp:
        json.dump(json_ready(summary), fp, indent=2)

    def fmt(value: Optional[float]) -> str:
        return "n/a" if value is None or not math.isfinite(value) else f"{value:.4f}"

    best_spear_name = "n/a" if best_spearman.empty else str(best_spearman.iloc[0]["score"])
    best_pr_name = "n/a" if best_pr.empty else str(best_pr.iloc[0]["score"])
    best_cap_name = "n/a" if best_captured.empty else str(best_captured.iloc[0]["score"])
    captured30 = metric_value(best_cap_name, "captured_pis_total_rank_at_0p3") if best_cap_name != "n/a" else None
    pr_best = metric_value(best_pr_name, "pr_auc_dangerous_total") if best_pr_name != "n/a" else None
    spear_best = metric_value(best_spear_name, "spearman_pis_total_rank") if best_spear_name != "n/a" else None
    fnr_base = metric_value("R_xwp_ode_vector", "fnr_dangerous_total_at_0p3")
    fnr_unc = metric_value("R_xwp_ode_vector_u_eta0p1", "fnr_dangerous_total_at_0p3")

    report_lines = [
        "# E6-D0 xWPCache Offline Diagnostic Report",
        "",
        f"Valid evaluated windows: {len(df)}",
        "",
        "## Best Scores",
        "",
        f"- Best Spearman vs pis_total_rank: `{best_spear_name}` ({fmt(spear_best)}).",
        f"- Best PR-AUC for dangerous_total: `{best_pr_name}` ({fmt(pr_best)}).",
        f"- Best CapturedPIS@30: `{best_cap_name}` ({fmt(captured30)}).",
        "",
        "## Required Questions",
        "",
        (
            f"- xWPCache vs time/raw/SEA baseline: Spearman "
            f"{'wins' if xwp_beats_baseline_spearman else 'does not win'} "
            f"({xwp_main} `{fmt(xwp_spear)}` vs baseline best `{fmt(baseline_best_spear)}`); "
            f"PR-AUC {'wins' if xwp_beats_baseline_pr else 'does not win'} "
            f"(`{fmt(xwp_pr)}` vs `{fmt(baseline_best_pr)}`); "
            f"CapturedPIS@30 {'wins' if xwp_beats_baseline_captured else 'does not win'} "
            f"(`{fmt(xwp_captured)}` vs `{fmt(baseline_best_captured)}`)."
        ),
        f"- Controlled correlation after time features: {xwp_main} LOSO residual Spearman `{fmt(xwp_controlled)}`.",
        f"- Full-xhat oracle vs Wiener proxy: oracle `{fmt(oracle_spear)}`, Wiener vector `{fmt(wiener_spear)}`.",
        f"- Vector vs scalar: vector `{fmt(vector_spear)}`, scalar `{fmt(scalar_spear)}`.",
        f"- Uncertainty effect: vector+eta0.1 Spearman `{fmt(uncertainty_spear)}`, base vector `{fmt(vector_spear)}`; FNR@30 base `{fmt(fnr_base)}`, eta0.1 `{fmt(fnr_unc)}`.",
        "",
        "If xWPCache does not beat time-only/raw/SEA here, treat that as a diagnostic failure or a proxy-quality problem before attempting online cache.",
        "",
        "## Files",
        "",
        f"- Metrics: `{out_dir / 'score_metrics.csv'}`",
        f"- Captured PIS: `{out_dir / 'captured_pis_at_budget.csv'}`",
        f"- False negatives: `{out_dir / 'failure_false_negative.csv'}`",
        f"- Plots: `{plots_dir}`",
    ]
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[E6-D0 eval] report written to {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
