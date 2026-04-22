#!/usr/bin/env python3
"""
ImageNet-256 evaluation wrapper using the official OpenAI ADM evaluation suite.

This script follows the evaluation path referenced by the PixelGen paper/README:
use OpenAI guided-diffusion's official `evaluations/evaluator.py` together with
the official ImageNet-256 reference batch `VIRTUAL_imagenet256_labeled.npz`.

What this script does:
1. Validates sample .npz files.
2. Downloads the official evaluator.py if needed.
3. Downloads the official ImageNet-256 reference batch if needed.
4. Runs the official evaluator as a subprocess.
5. Parses and saves Inception Score / FID / sFID / Precision / Recall.

Important:
- This is a thin wrapper around the official ADM evaluation code.
- The subprocess Python environment must have the official evaluator deps:
  tensorflow-gpu>=2.0 (or a compatible TF build), scipy, requests, tqdm.
  
  python eval_npz_imagenet_adm_official.py \
  --samples /mnt/iset/nfs-main/private/zhuangyifan/try/PixelGen/universal_pix_workdirs/exp_PixelGen_XL/val_ode50_cfg1.0/predict/output_shuffled_seed2026.npz

"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


OFFICIAL_EVALUATOR_URL = (
    "https://raw.githubusercontent.com/openai/guided-diffusion/main/evaluations/evaluator.py"
)
OFFICIAL_REF_BATCH_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/"
    "VIRTUAL_imagenet256_labeled.npz"
)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "adm_eval"
DEFAULT_EVALUATOR_PATH = DEFAULT_CACHE_DIR / "openai_guided_diffusion_evaluator.py"
DEFAULT_REF_NPZ_PATH = DEFAULT_CACHE_DIR / "VIRTUAL_imagenet256_labeled.npz"
DEFAULT_JSON_OUT = "imagenet256_metrics_adm_official.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        nargs="+",
        required=True,
        help="One or more .npz files to evaluate.",
    )
    parser.add_argument(
        "--ref-npz",
        default=str(DEFAULT_REF_NPZ_PATH),
        help="Path to the official ImageNet-256 reference batch. Downloaded if missing.",
    )
    parser.add_argument(
        "--evaluator-py",
        default=str(DEFAULT_EVALUATOR_PATH),
        help="Path to official OpenAI evaluator.py. Downloaded if missing.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Cache directory for evaluator artifacts.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the official evaluator subprocess.",
    )
    parser.add_argument(
        "--json-out",
        default=DEFAULT_JSON_OUT,
        help="Path to write parsed evaluation results as JSON.",
    )
    parser.add_argument(
        "--keep-stdout",
        action="store_true",
        help="Store the full evaluator stdout in the output JSON.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Fail instead of downloading missing official files.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dst: Path) -> None:
    ensure_parent(dst)
    print(f"[Download] {url}")
    print(f"[To] {dst}")
    with urllib.request.urlopen(url) as response, open(dst, "wb") as fp:
        fp.write(response.read())


def ensure_official_file(path: Path, url: str, skip_download: bool) -> None:
    if path.exists():
        return
    if skip_download:
        raise FileNotFoundError(f"Missing required file: {path}")
    download_file(url, path)


def validate_npz(npz_path: Path) -> Dict[str, object]:
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=False) as obj:
        if "arr_0" not in obj:
            raise KeyError(f"{npz_path} missing arr_0. Keys: {list(obj.keys())}")
        arr = obj["arr_0"]

    if arr.ndim != 4:
        raise ValueError(f"{npz_path} arr_0 must be 4D, got {arr.shape}")

    layout = "unknown"
    if arr.shape[-1] in (1, 3):
        layout = "NHWC"
    elif arr.shape[1] in (1, 3):
        layout = "NCHW"

    if layout != "NHWC":
        raise ValueError(
            f"{npz_path} arr_0 must be NHWC for the official ADM evaluator, got shape={arr.shape}"
        )

    if arr.dtype != np.uint8:
        print(f"[Warn] {npz_path} dtype is {arr.dtype}, official evaluator expects pixel values in [0, 255].")

    arr_min = int(arr.min())
    arr_max = int(arr.max())
    if arr_min < 0 or arr_max > 255:
        raise ValueError(
            f"{npz_path} arr_0 has values outside [0, 255]: min={arr_min}, max={arr_max}"
        )

    return {
        "sample_count": int(arr.shape[0]),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": arr_min,
        "max": arr_max,
    }


def parse_metrics(stdout: str) -> Dict[str, float]:
    metric_patterns = {
        "inception_score": r"Inception Score:\s*([-+eE0-9\.]+)",
        "fid": r"FID:\s*([-+eE0-9\.]+)",
        "sfid": r"sFID:\s*([-+eE0-9\.]+)",
        "precision": r"Precision:\s*([-+eE0-9\.]+)",
        "recall": r"Recall:\s*([-+eE0-9\.]+)",
    }

    metrics: Dict[str, float] = {}
    for key, pattern in metric_patterns.items():
        match = re.search(pattern, stdout)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def run_official_evaluator(
    python_exe: str,
    evaluator_py: Path,
    ref_npz: Path,
    sample_npz: Path,
    workdir: Path,
) -> subprocess.CompletedProcess:
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [python_exe, str(evaluator_py), str(ref_npz), str(sample_npz)]
    print(f"[Run] {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=str(workdir),
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    args = parse_args()

    cache_dir = Path(args.cache_dir)
    evaluator_py = Path(args.evaluator_py)
    ref_npz = Path(args.ref_npz)
    json_out = Path(args.json_out)
    workdir = cache_dir / "official_evaluator_workdir"

    ensure_official_file(evaluator_py, OFFICIAL_EVALUATOR_URL, args.skip_download)
    ensure_official_file(ref_npz, OFFICIAL_REF_BATCH_URL, args.skip_download)

    results: List[Dict[str, object]] = []
    for sample in args.samples:
        sample_path = Path(sample)
        print(f"\n[Validate] {sample_path}")
        sample_meta = validate_npz(sample_path)

        proc = run_official_evaluator(
            python_exe=args.python,
            evaluator_py=evaluator_py,
            ref_npz=ref_npz,
            sample_npz=sample_path,
            workdir=workdir,
        )

        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)

        item: Dict[str, object] = {
            "sample": str(sample_path),
            "reference_batch": str(ref_npz),
            "evaluator_py": str(evaluator_py),
            "official_source": "openai/guided-diffusion evaluations/evaluator.py",
            "sample_meta": sample_meta,
            "returncode": proc.returncode,
        }

        metrics = parse_metrics(proc.stdout)
        item.update(metrics)

        if args.keep_stdout:
            item["stdout"] = proc.stdout
            item["stderr"] = proc.stderr

        if proc.returncode != 0:
            item["error"] = (
                "Official evaluator subprocess failed. "
                "Make sure the selected Python has tensorflow/scipy/requests/tqdm installed."
            )

        results.append(item)

    with open(json_out, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[Saved] {json_out}")


if __name__ == "__main__":
    main()
